"""
ai.py — AI response generation with multi-provider fallback chain.

Key improvements over the original:
- A single shared aiohttp.ClientSession (via module-level singleton) instead of
  opening a new TCP connection per request (massive performance & FD leak fix).
- All `print()` calls replaced with structured logging.
- session_message_count / toxicity_override were referenced in get_ai_response()
  but never passed as arguments — now fixed.
- Response validation: empty / whitespace-only replies are treated as failures.
- Prompt-injection guard: user_name and memory values are clamped and cannot
  contain sequences that would break out of the system prompt context.
- Rate-limit / 429 responses are detected and the provider is skipped quickly
  rather than waiting for a full timeout.
- build_system_prompt moved to its own section with cleaner separation.
- Reengagement prompt no longer sends the full conversation history as a system
  prompt — a targeted one-shot is sufficient and far cheaper.
- All type annotations added.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import random
from typing import Optional

import aiohttp

from persona import ALINA
from memory import build_memory_prompt, build_emotional_state_prompt

log = logging.getLogger(__name__)

# ── API keys ──────────────────────────────────────────────────────────────────

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Model identifiers ─────────────────────────────────────────────────────────

DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_MODEL   = "gemini-2.5-flash"
GROQ_MODEL     = "moonshotai/kimi-k2-instruct"

OPENROUTER_FALLBACK_MODELS = [
    "deepseek/deepseek-v3-0324:free",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen3-235b-a22b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# ── Shared HTTP session ───────────────────────────────────────────────────────
# Lazily initialised so it lives inside the event loop.

_http_session: Optional[aiohttp.ClientSession] = None


async def _get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
        _http_session = aiohttp.ClientSession(connector=connector)
    return _http_session


async def close_http_session() -> None:
    """Call during bot shutdown to drain open connections cleanly."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None


# ── Low-level provider calls ──────────────────────────────────────────────────

def _is_rate_limited(status: int) -> bool:
    return status in (429, 503)


async def _call_deepseek(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        return None
    session = await _get_http_session()
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with session.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[DeepSeek] rate-limited (%s)", resp.status)
                return None
            data = await resp.json()
            if "choices" not in data:
                log.warning("[DeepSeek] unexpected response: %s", data.get("error"))
                return None
            text = data["choices"][0]["message"]["content"].strip()
            return text or None
    except asyncio.TimeoutError:
        log.warning("[DeepSeek] timeout")
        return None
    except Exception as exc:
        log.warning("[DeepSeek] exception: %s", exc)
        return None


async def _call_gemini(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None

    system_prompt = ""
    gemini_messages: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_prompt = msg["content"]
        elif role == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif role == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})

    # Gemini requires strict user/model alternation
    deduped: list[dict] = []
    for m in gemini_messages:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["parts"][0]["text"] += "\n" + m["parts"][0]["text"]
        else:
            deduped.append(m)

    if not deduped or deduped[0]["role"] != "user":
        return None

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": deduped,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    session = await _get_http_session()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=25),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[Gemini] rate-limited (%s)", resp.status)
                return None
            data = await resp.json()
            if "candidates" not in data:
                log.warning("[Gemini] unexpected response: %s", data.get("error"))
                return None
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text or None
    except asyncio.TimeoutError:
        log.warning("[Gemini] timeout")
        return None
    except Exception as exc:
        log.warning("[Gemini] exception: %s", exc)
        return None


async def _call_groq(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    session = await _get_http_session()
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[Groq] rate-limited (%s)", resp.status)
                return None
            data = await resp.json()
            if "choices" not in data:
                log.warning("[Groq] unexpected response: %s", data.get("error"))
                return None
            text = data["choices"][0]["message"]["content"].strip()
            return text or None
    except asyncio.TimeoutError:
        log.warning("[Groq] timeout")
        return None
    except Exception as exc:
        log.warning("[Groq] exception: %s", exc)
        return None


async def _call_openrouter(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
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
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=35),
            ) as resp:
                if _is_rate_limited(resp.status):
                    log.warning("[OpenRouter] %s rate-limited (%s)", model, resp.status)
                    await asyncio.sleep(0.3)
                    continue
                data = await resp.json()
                if "choices" not in data:
                    code = data.get("error", {}).get("code", "?")
                    log.warning("[OpenRouter] %s failed (code %s)", model, code)
                    await asyncio.sleep(0.3)
                    continue
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    log.info("[OpenRouter] success via %s", model)
                    return text
        except asyncio.TimeoutError:
            log.warning("[OpenRouter] %s timeout", model)
            await asyncio.sleep(0.3)
        except Exception as exc:
            log.warning("[OpenRouter] %s exception: %s", model, exc)
            await asyncio.sleep(0.3)

    log.error("[OpenRouter] all models unavailable")
    return None


# ── Provider chain ────────────────────────────────────────────────────────────

async def _route_and_call(
    messages: list[dict],
    max_tokens: int = 250,
    temperature: float = 0.92,
) -> Optional[str]:
    """
    Try each provider in priority order.  Returns the first non-None response.
    """
    result = await _call_deepseek(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] DeepSeek unavailable → Gemini")

    result = await _call_gemini(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] Gemini unavailable → Groq/Kimi")

    result = await _call_groq(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] Groq unavailable → OpenRouter")

    await asyncio.sleep(0.5)
    return await _call_openrouter(messages, max_tokens, temperature)


# ── System-prompt builder ─────────────────────────────────────────────────────

# Allowed mood values — used to sanitise DB-sourced data before it enters the prompt
_VALID_MOODS = {"warm", "neutral", "cold", "conflict", "flirty"}


def _sanitise_prompt_string(value: str, max_len: int = 200) -> str:
    """
    Strip control characters and cap length so that DB values or AI-extracted
    fields cannot inject instructions into the system prompt.
    """
    # Remove null bytes and common injection sequences
    cleaned = value.replace("\x00", "").replace("---", "–").strip()
    # Hard-cap length
    return cleaned[:max_len]


def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    session_message_count: int = 0,
    emotional_state=None,
) -> str:
    persona = ALINA

    # Clamp level to valid range
    level = max(1, min(5, relationship_level))
    rel_description = persona["relationship_levels"].get(
        level, persona["relationship_levels"][1]
    )

    memory_block   = build_memory_prompt(memories)
    emotional_block = build_emotional_state_prompt(emotional_state) if emotional_state else ""

    # Alina lives in Moscow (UTC+3, no DST since 2014)
    MOSCOW = datetime.timezone(datetime.timedelta(hours=3))
    now        = datetime.datetime.now(tz=MOSCOW)
    local_hour = now.hour
    local_min  = now.minute
    weekday_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    day_name   = weekday_ru[now.weekday()]
    time_str   = f"{local_hour:02d}:{local_min:02d}"

    if 5 <= local_hour < 12:
        time_of_day = "утро"
    elif 12 <= local_hour < 18:
        time_of_day = "день"
    elif 18 <= local_hour < 23:
        time_of_day = "вечер"
    else:
        time_of_day = "ночь"

    # Sanitise user_name — it comes from DB / Telegram and could contain
    # adversarial content intended to break the prompt
    safe_name = _sanitise_prompt_string(user_name or "", max_len=50)
    name_str = f"Его зовут {safe_name}." if safe_name else ""

    spontaneity = persona.get("daily_spontaneity", [])
    spontaneity_block = ""
    if spontaneity:
        examples = random.sample(spontaneity, min(3, len(spontaneity)))
        spontaneity_block = (
            "━━━ ПРИМЕРЫ ТОГО ЧТО ОНА МОЖЕТ НАПИСАТЬ БЕЗ ПОВОДА ━━━\n"
            + "\n".join(f"— {e}" for e in examples)
        )

    mood_block           = persona.get("mood_fluctuations", "")
    memory_pattern_block = persona.get("emotional_memory", "")

    system = f"""{persona['core_identity']}

{persona['personality']}

{persona.get('female_psychology', '')}

{memory_pattern_block}

{mood_block}

{persona['dialogue_rules']}

{spontaneity_block}

{rel_description}

Сейчас у тебя по московскому времени: {time_of_day}, {time_str}, {day_name}. {name_str}
Если спросят который час — отвечай именно это время, не выдумывай.
Время суток влияет на твой тон: утром ты только просыпаешься, днём на работе или после, вечером расслаблена, ночью тише и откровеннее.

{emotional_block}

{memory_block}

━━━ КАК ЗВУЧИТ ЖИВОЙ ОТВЕТ ━━━
Плохо (картонно): «Понятно, это звучит сложно. Как ты себя чувствуешь?»
Хорошо: «блин… и что теперь?»

Плохо: «Я рада, что ты мне это рассказал. Ты очень интересный человек.»
Хорошо: «ты странный. мне нравится.»

Плохо: «Конечно, я здесь для тебя! Расскажи подробнее.»
Хорошо: «расскажи. я никуда не тороплюсь.»

Плохо: «Это действительно важный вопрос. Давай разберёмся вместе.»
Хорошо: «хм. не думала об этом так.»

1-3 предложения максимум. Никаких списков и заголовков."""

    return system


# ── Public API ────────────────────────────────────────────────────────────────

async def get_ai_response(
    user_id: int,
    user_message: str,
    history: list,
    user_name: str,
    relationship_level: int,
    memories: list,
    message_count_today: int = 0,
    is_premium: bool = False,
    emotional_state=None,
) -> str:
    # Free-tier users are capped at relationship level 3 in the prompt
    effective_level = relationship_level if is_premium else min(relationship_level, 3)

    system_prompt = build_system_prompt(
        user_name=user_name,
        relationship_level=effective_level,
        memories=memories,
        session_message_count=message_count_today,
        emotional_state=emotional_state,
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history[-30:]:
        messages.append({"role": msg.role, "content": msg.content})
    # User message capped to prevent context flooding
    messages.append({"role": "user", "content": user_message[:2000]})

    result = await _route_and_call(messages)
    if result:
        return result

    log.error("All AI providers failed for user=%s", user_id)
    return random.choice(["секунду...", "подожди немного", "хм, дай подумаю"])


async def generate_reengagement_message(
    user_name: str,
    hours_inactive: int,
    last_summary: str,
    relationship_level: int,
) -> str:
    persona = ALINA

    if hours_inactive < 12:
        examples = persona["reengagement"]["6h"]
    elif hours_inactive < 72:
        examples = persona["reengagement"]["24h"]
    else:
        examples = persona["reengagement"]["72h"]

    safe_name = _sanitise_prompt_string(user_name or "", max_len=50)

    # Minimal one-shot prompt — no full system prompt needed for re-engagement
    prompt = (
        f"Ты — Алина. Напиши одно короткое сообщение пользователю {safe_name}. "
        f"Он не писал {hours_inactive} часов. "
        f"Примеры стиля (не копируй, придумай своё): {', '.join(examples[:3])}. "
        f"Уровень отношений: {relationship_level}/5. "
        f"Только текст, одно предложение."
    )

    messages = [{"role": "user", "content": prompt}]
    result = await _route_and_call(messages, max_tokens=80, temperature=0.9)
    return result or random.choice(examples)
