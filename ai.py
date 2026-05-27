"""
ai.py — Генерация AI-ответов с цепочкой провайдеров. v2.1 (hotfix).

Исправлен циклический импорт:
  ai.py → memory.py → ai.py  (ImportError при старте)

Решение: HTTP-сессия вынесена в http_client.py.
  ai.py     → http_client.py  (нет петли)
  memory.py → http_client.py  (нет петли)

Импорт memory (build_memory_prompt, build_emotional_state_prompt) перенесён
внутрь функции build_system_prompt — на момент вызова оба модуля уже
полностью загружены.
"""

from __future__ import annotations

import datetime
import logging
import os
import random
from typing import Optional

from persona import ALINA
from http_client import get_http_session, close_http_session  # noqa: F401 (re-export)

log = logging.getLogger(__name__)

# ── API ключи ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Модели ────────────────────────────────────────────────────────────────────

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


# ── Определение rate-limiting ─────────────────────────────────────────────────

def _is_rate_limited(status: int) -> bool:
    return status in (429, 503)


# ── Вызовы провайдеров ────────────────────────────────────────────────────────

async def _call_deepseek(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        return None
    session = await get_http_session()
    try:
        async with session.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[DeepSeek] rate-limited (%s)", resp.status)
                return None
            data = await resp.json()
            if "choices" not in data:
                log.warning("[DeepSeek] неожиданный ответ: %s", data.get("error"))
                return None
            text = data["choices"][0]["message"]["content"].strip()
            return text or None
    except asyncio.TimeoutError:
        log.warning("[DeepSeek] timeout")
        return None
    except Exception as exc:
        log.warning("[DeepSeek] исключение: %s", exc)
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

    # Gemini требует строгого чередования user/model
    deduped: list[dict] = []
    for m in gemini_messages:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["parts"][0]["text"] += "\n" + m["parts"][0]["text"]
        else:
            deduped.append(m)

    if not deduped or deduped[0]["role"] != "user":
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    session = await get_http_session()
    try:
        async with session.post(
            url,
            # API-ключ в заголовке, а не в URL — не попадает в логи запросов
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": deduped,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=aiohttp.ClientTimeout(total=25),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[Gemini] rate-limited (%s)", resp.status)
                return None
            data = await resp.json()
            if "candidates" not in data:
                log.warning("[Gemini] неожиданный ответ: %s", data.get("error"))
                return None
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text or None
    except asyncio.TimeoutError:
        log.warning("[Gemini] timeout")
        return None
    except Exception as exc:
        log.warning("[Gemini] исключение: %s", exc)
        return None


async def _call_groq(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    session = await get_http_session()
    try:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[Groq] rate-limited (%s)", resp.status)
                return None
            data = await resp.json()
            if "choices" not in data:
                log.warning("[Groq] неожиданный ответ: %s", data.get("error"))
                return None
            text = data["choices"][0]["message"]["content"].strip()
            return text or None
    except asyncio.TimeoutError:
        log.warning("[Groq] timeout")
        return None
    except Exception as exc:
        log.warning("[Groq] исключение: %s", exc)
        return None


async def _call_openrouter(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    session = await get_http_session()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alina-bot",
        "X-Title": "Alina Bot",
    }
    for model in OPENROUTER_FALLBACK_MODELS:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
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
                    log.info("[OpenRouter] успех: %s", model)
                    return text
        except asyncio.TimeoutError:
            log.warning("[OpenRouter] %s timeout", model)
            await asyncio.sleep(0.3)
        except Exception as exc:
            log.warning("[OpenRouter] %s исключение: %s", model, exc)
            await asyncio.sleep(0.3)

    log.error("[OpenRouter] все модели недоступны")
    return None


# ── Цепочка провайдеров ───────────────────────────────────────────────────────

async def _route_and_call(
    messages: list[dict],
    max_tokens: int = 250,
    temperature: float = 0.92,
) -> Optional[str]:
    result = await _call_deepseek(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] DeepSeek недоступен → Gemini")

    result = await _call_gemini(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] Gemini недоступен → Groq/Kimi")

    result = await _call_groq(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] Groq недоступен → OpenRouter")

    await asyncio.sleep(0.5)
    return await _call_openrouter(messages, max_tokens, temperature)


# ── Санитизация строк для промпта ─────────────────────────────────────────────

def _sanitise_prompt_string(value: str, max_len: int = 200) -> str:
    """
    Очищает строку перед вставкой в системный промпт.

    Угрозы:
    - Переносы строк в имени (Telegram first_name может содержать \n)
    - "SYSTEM:", "Инструкция:", "━━━" — инъекция директив
    - Нулевые байты
    - Управляющие символы

    Стратегия: приводим к однострочному тексту, нейтрализуем маркеры.
    """
    if not value:
        return ""
    # Удаляем нулевые байты и управляющие символы (кроме пробела)
    cleaned = "".join(
        ch if ch >= " " or ch in ("\t",) else " "
        for ch in value
        if ch != "\x00"
    )
    # Переносы строк → пробел (имена из Telegram могут содержать \n)
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    # Нейтрализуем паттерны инъекции директив
    for marker in ("SYSTEM:", "Инструкция:", "system:", "━━━", "───", "---"):
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()[:max_len]


# ── Системный промпт ──────────────────────────────────────────────────────────

def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    session_message_count: int = 0,
    emotional_state=None,
) -> str:
    # Ленивый импорт: к моменту вызова оба модуля уже полностью загружены.
    # На уровне модуля импортировать нельзя — circular import (ai <-> memory).
    from memory import build_memory_prompt, build_emotional_state_prompt

    persona = ALINA

    level = max(1, min(5, relationship_level))
    rel_description = persona["relationship_levels"].get(
        level, persona["relationship_levels"][1]
    )

    memory_block    = build_memory_prompt(memories)
    emotional_block = build_emotional_state_prompt(emotional_state) if emotional_state else ""

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

    safe_name = _sanitise_prompt_string(user_name or "", max_len=50)
    name_str  = f"Его зовут {safe_name}." if safe_name else ""

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

    # .get() с fallback — KeyError больше невозможен даже если persona изменится
    dialogue_rules = persona.get("dialogue_rules", "")

    system = f"""{persona['core_identity']}

{persona['personality']}

{persona.get('female_psychology', '')}

{memory_pattern_block}

{mood_block}

{dialogue_rules}

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

    # Логируем приблизительный размер промпта для мониторинга токенов
    prompt_chars = len(system)
    if prompt_chars > 12000:
        log.warning(
            "Системный промпт очень большой: %d символов (~%d токенов) для user",
            prompt_chars, prompt_chars // 4,
        )

    return system


def _history_to_messages(history: list, max_content_per_msg: int = 1000) -> list[dict]:
    """
    Конвертирует историю ORM-объектов в список dict для API.
    Усекает каждое сообщение до max_content_per_msg символов —
    защита от накопления огромных сообщений в истории (context flooding).
    """
    result = []
    for msg in history[-30:]:
        content = msg.content
        if len(content) > max_content_per_msg:
            content = content[:max_content_per_msg] + "…"
        result.append({"role": msg.role, "content": content})
    return result


# ── Публичный API ─────────────────────────────────────────────────────────────

# Sentinel — означает что все провайдеры упали
FALLBACK_RESPONSES = ["секунду...", "подожди немного", "хм, дай подумаю"]


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
) -> tuple[str, bool]:
    """
    Возвращает (response_text, is_fallback).
    is_fallback=True означает что все провайдеры упали — ответ не надо
    сохранять в историю и не надо считать за "реальный обмен".
    """
    effective_level = relationship_level if is_premium else min(relationship_level, 3)

    system_prompt = build_system_prompt(
        user_name=user_name,
        relationship_level=effective_level,
        memories=memories,
        session_message_count=message_count_today,
        emotional_state=emotional_state,
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_messages(history))
    messages.append({"role": "user", "content": user_message[:2000]})

    result = await _route_and_call(messages)
    if result:
        return result, False

    log.error("Все AI-провайдеры недоступны для user=%s", user_id)
    return random.choice(FALLBACK_RESPONSES), True


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
