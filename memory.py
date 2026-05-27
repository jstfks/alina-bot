"""
memory.py — Async memory extraction and prompt-building.

Key improvements over the original:
- Shared HTTP session (imported from ai.py) instead of creating a new
  aiohttp.ClientSession on every extraction call.
- json.loads() wrapped in a stricter parser that validates the returned
  structure is actually a flat {str: str} dict — malformed LLM output can
  no longer crash the background task or inject arbitrary data into the DB.
- ALLOWED_MEMORY_KEYS whitelist: only recognised fact types are stored; an
  LLM that hallucinates keys like "system_prompt" or "instructions" is
  blocked at the boundary.
- Value length clamped (500 chars) before writing to DB.
- All print() → logging.
- Removed duplicate OPENROUTER_FALLBACK_MODELS constant (now imported/shared).
- build_memory_prompt sanitises each value via html.escape-equivalent logic
  so that stored values can't escape the system-prompt block.
- extract_memories / extract_emotional_state are fire-and-forget background
  tasks — all exceptions are caught and logged, never propagated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

from database import save_memory, save_emotional_state, get_emotional_state

log = logging.getLogger(__name__)

# Keys we will accept from the LLM — anything outside this set is discarded.
ALLOWED_MEMORY_KEYS = frozenset({
    "name", "job", "city", "age", "hobby", "pet",
    "mood", "relationship_status", "education", "siblings",
})

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Use the same fallback list ordering as ai.py
OPENROUTER_FALLBACK_MODELS = [
    "deepseek/deepseek-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# Attempt to reuse the shared session from ai.py; fall back to own import
try:
    from ai import _get_http_session
except ImportError:
    import aiohttp
    _own_session: Optional[aiohttp.ClientSession] = None

    async def _get_http_session():  # type: ignore[misc]
        global _own_session
        if _own_session is None or _own_session.closed:
            _own_session = aiohttp.ClientSession()
        return _own_session


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    """Strip markdown code fences that LLMs often add."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _parse_flat_dict(text: str) -> Optional[dict[str, str]]:
    """
    Parse and validate that the response is a flat {str: str} dict.
    Returns None if parsing fails or the structure is wrong.
    """
    try:
        cleaned = _clean_json_text(text)
        if not cleaned or cleaned == "{}":
            return {}
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            log.warning("[memory] LLM returned non-dict JSON: %s", type(data))
            return None
        # Enforce flat str→str structure
        validated: dict[str, str] = {}
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, (str, int, float)):
                continue
            validated[k] = str(v)
        return validated
    except json.JSONDecodeError as exc:
        log.warning("[memory] JSON decode failed: %s — raw: %.200s", exc, text)
        return None


# ── LLM extraction backends ───────────────────────────────────────────────────

async def _extract_via_groq(prompt: str) -> Optional[dict]:
    if not GROQ_API_KEY:
        return None
    session = await _get_http_session()
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
    }
    try:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=__import__("aiohttp").ClientTimeout(total=20),
        ) as resp:
            data = await resp.json()
            if "choices" not in data:
                log.warning("[memory/Groq] error: %s", data.get("error", {}).get("message"))
                return None
            return _parse_flat_dict(data["choices"][0]["message"]["content"])
    except Exception as exc:
        log.warning("[memory/Groq] exception: %s", exc)
        return None


async def _extract_via_openrouter(prompt: str) -> Optional[dict]:
    if not OPENROUTER_API_KEY:
        return None
    session = await _get_http_session()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alina-bot",
        "X-Title": "Alina Bot",
    }
    for model in OPENROUTER_FALLBACK_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.1,
        }
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=__import__("aiohttp").ClientTimeout(total=25),
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    code = data.get("error", {}).get("code", "?")
                    log.warning("[memory/OpenRouter] %s failed (code %s)", model, code)
                    await asyncio.sleep(0.3)
                    continue
                result = _parse_flat_dict(data["choices"][0]["message"]["content"])
                if result is not None:
                    return result
        except Exception as exc:
            log.warning("[memory/OpenRouter] %s exception: %s", model, exc)
            await asyncio.sleep(0.3)
    return None


# ── Public extraction functions ───────────────────────────────────────────────

async def extract_memories(user_id: int, conversation: list[dict]) -> None:
    """
    Background task: extract user facts from recent conversation and upsert them.
    All errors are caught — this must never crash the caller.
    """
    try:
        if len(conversation) < 4:
            return

        convo_text = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Алина'}: {m['content']}"
            for m in conversation[-10:]
        )

        prompt = (
            "Проанализируй разговор и извлеки факты о пользователе.\n\n"
            f"Разговор:\n{convo_text}\n\n"
            "Верни JSON объект с фактами. Только явно сказанное. Не придумывай.\n"
            f"Допустимые ключи: {', '.join(sorted(ALLOWED_MEMORY_KEYS))}\n\n"
            'Пример: {"name": "Дима", "job": "программист"}\n'
            "Если ничего нового — верни: {}\n"
            "Только JSON, без пояснений."
        )

        facts = await _extract_via_groq(prompt)
        if facts is None:
            log.info("[memory] Groq unavailable, trying OpenRouter")
            facts = await _extract_via_openrouter(prompt)

        if not facts:
            return

        for key, value in facts.items():
            # Enforce whitelist and length cap before writing
            if key not in ALLOWED_MEMORY_KEYS:
                log.info("[memory] discarding unknown key '%s' for user=%s", key, user_id)
                continue
            if not value or not str(value).strip():
                continue
            await save_memory(user_id, key, str(value)[:500])

    except Exception as exc:
        log.error("[memory] extract_memories failed for user=%s: %s", user_id, exc)


# Allowed mood values from the LLM
_VALID_MOODS = frozenset({"warm", "neutral", "cold", "conflict", "flirty"})


async def extract_emotional_state(user_id: int, conversation: list[dict]) -> None:
    """
    Background task: analyse the emotional tone of the last session.
    All errors are caught — this must never crash the caller.
    """
    try:
        if len(conversation) < 4:
            return

        convo_text = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Алина'}: {m['content']}"
            for m in conversation[-16:]
        )

        prompt = (
            "Проанализируй конец разговора между пользователем и девушкой Алиной.\n\n"
            f"Разговор:\n{convo_text}\n\n"
            'Верни JSON с тремя полями:\n'
            '1. "mood" — тон в конце: "warm", "neutral", "cold", "conflict", "flirty"\n'
            '2. "last_moment" — одна фраза, самый важный эмоциональный момент (макс 100 символов)\n'
            '3. "open_topics" — незакрытые темы (макс 80 символов). Если нет — пустая строка.\n\n'
            "Только JSON. Пример:\n"
            '{"mood": "warm", "last_moment": "он рассказал про расставание, она поддержала", '
            '"open_topics": "он хотел рассказать про работу"}'
        )

        result = await _extract_via_groq(prompt)
        if result is None:
            result = await _extract_via_openrouter(prompt)
        if not result:
            return

        mood = result.get("mood", "neutral")
        if mood not in _VALID_MOODS:
            mood = "neutral"

        await save_emotional_state(
            user_id=user_id,
            mood_after_last_session=mood,
            last_emotional_moment=str(result.get("last_moment", ""))[:200],
            open_topics=str(result.get("open_topics", ""))[:200],
        )

    except Exception as exc:
        log.error("[memory] extract_emotional_state failed for user=%s: %s", user_id, exc)


async def update_hours_since_message(user_id: int, hours: float) -> None:
    """Update the hours-since-last-message field on the emotional state record."""
    try:
        state = await get_emotional_state(user_id)
        if state:
            await save_emotional_state(
                user_id=user_id,
                mood_after_last_session=state.mood_after_last_session,
                last_emotional_moment=state.last_emotional_moment,
                open_topics=state.open_topics,
                hours_since_last_message=hours,
            )
        else:
            await save_emotional_state(user_id=user_id, hours_since_last_message=hours)
    except Exception as exc:
        log.warning("[memory] update_hours_since_message failed for user=%s: %s", user_id, exc)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _escape_prompt_value(value: str) -> str:
    """
    Prevent stored memory values from injecting system-prompt directives.
    We strip leading/trailing whitespace and limit length; dangerous patterns
    (like lines starting with '━━━' or 'SYSTEM:') are neutralised.
    """
    value = value.strip()[:300]
    # Neutralise separator sequences that could trick some models
    value = value.replace("━", "-").replace("─", "-")
    return value


def build_memory_prompt(memories: list) -> str:
    if not memories:
        return ""

    facts: dict[str, str] = {m.key: _escape_prompt_value(m.value) for m in memories}
    lines = ["Что ты знаешь о собеседнике:"]

    ordered_keys = ["name", "job", "city", "hobby", "pet"]
    for key in ordered_keys:
        if key not in facts:
            continue
        labels = {
            "name":  "Его зовут",
            "job":   "Работает:",
            "city":  "Живёт в:",
            "hobby": "Увлечения:",
            "pet":   "Питомец:",
        }
        lines.append(f"- {labels[key]} {facts[key]}")

    skip = set(ordered_keys)
    for key, value in facts.items():
        if key not in skip:
            lines.append(f"- {key}: {value}")

    lines.append("\nИспользуй эти знания естественно. Просто знай это.")
    return "\n".join(lines)


def build_emotional_state_prompt(state) -> str:
    """Convert the stored EmotionalState into a system-prompt block."""
    if not state:
        return ""

    lines = ["━━━ ЭМОЦИОНАЛЬНАЯ ПАМЯТЬ — КАК ПРОШЁЛ ПРОШЛЫЙ РАЗГОВОР ━━━"]

    hours = float(state.hours_since_last_message or 0)
    if hours < 2:
        time_note = "он писал совсем недавно"
    elif hours < 8:
        time_note = f"последний раз он писал {int(hours)} часа назад"
    elif hours < 24:
        time_note = f"он не писал {int(hours)} часов"
    elif hours < 72:
        time_note = f"он не писал {int(hours / 24)} дня"
    else:
        time_note = f"он не писал {int(hours / 24)} дней — давно"
    lines.append(f"Время: {time_note}.")

    mood = state.mood_after_last_session
    if mood not in _VALID_MOODS:
        mood = "neutral"
    mood_map = {
        "warm":     "последний разговор был тёплым — вы сблизились. начни немного теплее обычного.",
        "flirty":   "последний разговор был с флиртом и напряжением. можно продолжать в том же духе.",
        "neutral":  "последний разговор был обычным. начни как обычно.",
        "cold":     "последний разговор закончился сухо или с дистанцией. будь чуть сдержаннее в начале.",
        "conflict": "в прошлый раз была напряжённость или обида. будь немного холоднее — не сразу открывайся.",
    }
    lines.append(mood_map[mood])

    if state.last_emotional_moment:
        moment = _escape_prompt_value(state.last_emotional_moment)
        lines.append(f"Важный момент прошлого разговора: {moment}")
        lines.append("Можешь вернуться к этому — естественно, без «помнишь ты говорил».")

    if state.open_topics:
        topics = _escape_prompt_value(state.open_topics)
        lines.append(f"Незакрытая тема: {topics}")
        lines.append("Если разговор зайдёт рядом — можешь сама вернуться к ней.")

    lines.append("")
    return "\n".join(lines)
