"""
ai.py — Генерация AI-ответов с цепочкой провайдеров. v3.0

Изменения v3.0:
  - Переход на слоистую персону (persona/)
  - from persona import ALINA → from persona import CORE_PROMPT, build_context_layers
  - _nsfw_block() удалена — переехала в persona/layers.py
  - build_system_prompt переработан: промпт уровня 1 ~790 токенов вместо ~2800
  - Сигнатуры всех публичных функций не изменились

Исправлен циклический импорт (v2.1):
  ai.py → memory.py → ai.py  (ImportError при старте)
  Решение: HTTP-сессия вынесена в http_client.py.
  Импорт memory внутри build_system_prompt — на момент вызова оба модуля загружены.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import random
from typing import Optional

import aiohttp

from persona import CORE_PROMPT, build_context_layers
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
    "openrouter/owl-alpha",                          # приоритет — stealth frontier
    "nousresearch/hermes-3-llama-3.1-405b:free",     # 405B, отлично держит персонажей + nsfw
    "z-ai/glm-4.5-air:free",                         # GLM 4.5 Air — сильный на русском
    "stepfun/step-3.5-flash:free",                   # Step Flash — быстрый
    "tngtech/deepseek-r1t2-chimera:free",            # 671B MoE, генерирует <think> — стрипаем
    "openai/gpt-oss-120b:free",                      # 117B MoE, сильный но с content filters
    "google/gemma-4-26b-a4b-it:free",                # Gemma 4 26B MoE — резерв
]

VISION_FALLBACK_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",           # приоритет 1 — Qwen VL 72B
    "qwen/qwen2.5-vl-32b-instruct",                # приоритет 2 — Qwen VL 32B
    "google/gemma-4-31b-it:free",                  # приоритет 3 — Gemma 4 31B
    "mistralai/pixtral-12b",                       # приоритет 4 — Pixtral 12B
    "google/gemma-4-26b-a4b-it:free",              # резерв
    "moonshotai/kimi-k2.6:free",                   # резерв
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # резерв
    "nvidia/nemotron-nano-12b-v2-vl:free",         # резерв
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
                err = data.get("error", {})
                if "Insufficient Balance" in str(err):
                    log.error("[DeepSeek] БАЛАНС КОНЧИЛСЯ — пополните счёт на platform.deepseek.com")
                else:
                    log.warning("[DeepSeek] неожиданный ответ: %s", err)
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


def _strip_think_tags(text: str) -> str:
    """
    Убирает <think>...</think> блоки из ответа.
    Qwen3 и некоторые другие модели добавляют их когда "думают вслух".
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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
                    "top_p": 0.92,
                },
                timeout=aiohttp.ClientTimeout(total=35),
            ) as resp:
                if _is_rate_limited(resp.status):
                    log.warning("[OpenRouter] %s rate-limited (%s)", model, resp.status)
                    await asyncio.sleep(0.3)
                    continue
                data = await resp.json()
                if "choices" not in data:
                    err  = data.get("error", {})
                    code = err.get("code", resp.status)
                    msg  = str(err.get("message", ""))[:120]
                    if resp.status == 404:
                        log.warning("[OpenRouter] %s — модель не найдена (снята?): %s", model, msg)
                    else:
                        log.warning("[OpenRouter] %s — ошибка %s: %s", model, code, msg)
                    await asyncio.sleep(0.3)
                    continue
                text = data["choices"][0]["message"]["content"].strip()
                text = _strip_think_tags(text)
                if text:
                    log.info("[OpenRouter] успех: %s", model)
                    return text
                log.warning("[OpenRouter] %s — пустой ответ после обрезки think-блоков", model)
                await asyncio.sleep(0.3)
                continue
        except asyncio.TimeoutError:
            log.warning("[OpenRouter] %s timeout", model)
            await asyncio.sleep(0.3)
        except Exception as exc:
            log.warning("[OpenRouter] %s исключение: %s", model, exc)
            await asyncio.sleep(0.3)

    log.error("[OpenRouter] все модели недоступны")
    return None


# ── Vision pipeline ───────────────────────────────────────────────────────────

async def get_image_description(
    image_b64: str,
    mime_type: str,
    caption: str,
) -> Optional[str]:
    """
    Шаг 1: vision-модель описывает изображение фактически.
    Возвращает текстовое описание для передачи в основную модель.
    """
    if not OPENROUTER_API_KEY:
        return None

    system = (
        "You are an image analysis assistant. "
        "Describe what you see in the image factually and concisely in Russian. "
        "Focus on: main subjects, actions, mood, setting, notable details. "
        "Max 3-4 sentences. No opinions, just facts."
    )
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
    ]
    if caption:
        content.append({"type": "text", "text": f"Подпись к фото: {caption}"})

    session = await get_http_session()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alina-bot",
        "X-Title": "Alina Bot",
    }
    for model in VISION_FALLBACK_MODELS:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": content},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
                timeout=aiohttp.ClientTimeout(total=40),
            ) as resp:
                if resp.status == 429:
                    log.warning("[Vision describe] %s rate-limited", model)
                    await asyncio.sleep(0.3)
                    continue
                data = await resp.json()
                if "choices" not in data:
                    log.warning("[Vision describe] %s ошибка: %s", model, data.get("error"))
                    await asyncio.sleep(0.3)
                    continue
                description = _strip_think_tags(
                    data["choices"][0]["message"]["content"].strip()
                )
                if description:
                    log.info("[Vision describe] успех: %s", model)
                    return description
        except asyncio.TimeoutError:
            log.warning("[Vision describe] %s timeout", model)
        except Exception as exc:
            log.warning("[Vision describe] %s исключение: %s", model, exc)
        await asyncio.sleep(0.3)

    # Gemini как fallback для описания
    try:
        gemini_msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ]},
        ]
        result = await _call_gemini(gemini_msgs, max_tokens=300, temperature=0.3)
        if result:
            return result
    except Exception as exc:
        log.warning("[Vision describe] Gemini fallback исключение: %s", exc)

    return None


async def get_ai_response_image(
    user_id: int,
    image_b64: str,
    mime_type: str,
    caption: str,
    user_name: str,
    relationship_level: int,
    memories: list,
    history: list,
    is_premium: bool = False,
    emotional_state=None,
    hours_since_last: float = 0.0,
    message_count_today: int = 0,
) -> tuple[str, bool]:
    """
    Двухэтапный pipeline для фото:
    1. Vision-модель описывает изображение фактически
    2. Основная модель отвечает голосом Алины
    """
    description = await get_image_description(image_b64, mime_type, caption)

    if description:
        if caption:
            user_message = f"[прислал фото] {caption}\n\nна фото: {description}"
        else:
            user_message = f"[прислал фото]\n\nна фото: {description}"
        log.info("[Image pipeline] описание получено, передаём в основную модель")
    else:
        log.warning("[Image pipeline] vision недоступна для user=%s, реагируем на факт фото", user_id)
        user_message = caption if caption else "[прислал фото, но описание недоступно]"

    return await get_ai_response(
        user_id=user_id,
        user_message=user_message,
        history=history,
        user_name=user_name,
        relationship_level=relationship_level,
        memories=memories,
        message_count_today=message_count_today,
        is_premium=is_premium,
        emotional_state=emotional_state,
        hours_since_last=hours_since_last,
    )


# ── Цепочка провайдеров ───────────────────────────────────────────────────────

async def _route_and_call(
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.92,
) -> Optional[str]:
    # 1. OpenRouter (7 моделей, Owl Alpha первым)
    result = await _call_openrouter(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] OpenRouter недоступен → Gemini")

    # 2. Gemini
    result = await _call_gemini(messages, max_tokens, temperature)
    if result:
        return result
    log.info("[route] Gemini недоступен → DeepSeek")

    # 3. DeepSeek
    await asyncio.sleep(0.5)
    return await _call_deepseek(messages, max_tokens, temperature)


# ── Тиеры сессии по времени неактивности ─────────────────────────────────────

def _session_tier(hours: float) -> tuple[int, str]:
    """
    Возвращает (history_limit, gap_prompt_block) по времени с последнего сообщения.

    Тиеры:
      < 2ч   — обычное продолжение, полная история (30 сообщений)
      2–10ч  — небольшой перерыв, чуть меньше контекста (20)
      10–24ч — новая сессия, только хвост переписки (6)
      24–72ч — почти с чистого листа (2)
      72ч+   — полный сброс, свежий старт (0)
    """
    h = int(hours)
    if hours < 2:
        return 30, ""
    if hours < 10:
        return 20, (
            "━━━ КОНТЕКСТ СЕССИИ ━━━\n"
            f"Между сообщениями прошло около {h} ч. Продолжай разговор естественно — "
            "БЕЗ приветствий, БЕЗ «привет» и «здравствуй»."
        )
    if hours < 24:
        return 6, (
            "━━━ КОНТЕКСТ СЕССИИ ━━━\n"
            f"Пауза {h} ч. Продолжай тепло и естественно — "
            "НЕ здоровайся, не говори «привет», просто продолжай общение. "
            "Рада что написал — скажи об этом своими словами, без формального приветствия."
        )
    if hours < 72:
        return 2, (
            "━━━ КОНТЕКСТ СЕССИИ ━━━\n"
            f"Пауза больше суток ({h} ч). Рада что вернулся — "
            "скажи это тепло, но БЕЗ слова «привет». Начни с ощущения, не с приветствия."
        )
    return 0, (
        "━━━ КОНТЕКСТ СЕССИИ ━━━\n"
        f"Очень долго не писал ({h} ч). Соскучилась — "
        "вырази это своими словами, НЕ говори «привет» или «здравствуй»."
    )


# ── Дуга сессии ───────────────────────────────────────────────────────────────

def build_session_arc(session_message_count: int) -> tuple[float, str]:
    """
    Возвращает (temperature, arc_text).

    BOLT+ рекомендует 0.70–0.85. Используем этот диапазон:
      1–3   → 0.72  ранняя стадия, стабильный образ
      4–10  → 0.78  разогрев, чуть свободнее
      11–20 → 0.82  глубокий разговор, живее
      20+   → 0.85  максимум по BOLT+
    """
    if session_message_count <= 3:
        return 0.72, (
            "━━━ ДУГА РАЗГОВОРА ━━━\n"
            "Ранняя стадия. Держи образ стабильно. Не форсируй тепло. Не здоровайся."
        )
    if session_message_count <= 10:
        return 0.78, (
            "━━━ ДУГА РАЗГОВОРА ━━━\n"
            "Разговор набирает глубину. Чуть свободнее — можно дать что-то личное."
        )
    if session_message_count <= 20:
        return 0.82, (
            "━━━ ДУГА РАЗГОВОРА ━━━\n"
            "Глубокий разговор. Расслаблена — может вырваться что-то настоящее."
        )
    return 0.85, (
        "━━━ ДУГА РАЗГОВОРА ━━━\n"
        "Долгий разговор. Открытая. Прямая. Без фильтра."
    )


# ── Санитизация строк для промпта ─────────────────────────────────────────────

def _sanitise_prompt_string(value: str, max_len: int = 200) -> str:
    """
    Очищает строку перед вставкой в системный промпт.
    """
    if not value:
        return ""
    cleaned = "".join(
        ch if ch >= " " or ch in ("\t",) else " "
        for ch in value
        if ch != "\x00"
    )
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    for marker in ("SYSTEM:", "Инструкция:", "system:", "━━━", "───", "---"):
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()[:max_len]


# ── Системный промпт ──────────────────────────────────────────────────────────

# Спонтанные фразы Алины (seed для тона в reengagement и spontaneity_block)
_SPONTANEITY = [
    "видела голубя, который украл хлеб у другого. долго думала.",
    "дочитала книгу в три ночи. теперь не сплю.",
    "купила цветы себе. не потому что грустно — просто захотела.",
    "самая популярная ложь — «я в порядке». на втором месте — «скоро».",
    "три часа ночи — всё кажется важным и бессмысленным одновременно.",
    "хорошо быть книгой. тебя либо читают, либо откладывают.",
]


def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    session_message_count: int = 0,
    emotional_state=None,
    arc_block: str = "",
    gap_block: str = "",
    nsfw_block: str = "",  # сохранён для обратной совместимости сигнатуры, не используется
) -> str:
    # Ленивый импорт: к моменту вызова оба модуля уже полностью загружены.
    # На уровне модуля импортировать нельзя — circular import (ai <-> memory).
    from memory import build_memory_prompt, build_emotional_state_prompt

    level = max(1, min(5, relationship_level))

    # ── Время (МСК) ───────────────────────────────────────────────────────────
    MOSCOW     = datetime.timezone(datetime.timedelta(hours=3))
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

    # ── Блоки из memory.py ────────────────────────────────────────────────────
    memory_block    = build_memory_prompt(memories)
    emotional_block = build_emotional_state_prompt(emotional_state) if emotional_state else ""

    # ── Спонтанность: 3 случайных примера ────────────────────────────────────
    spontaneity_block = (
        "━━━ ПРИМЕРЫ ТОГО ЧТО ОНА МОЖЕТ НАПИСАТЬ БЕЗ ПОВОДА ━━━\n"
        + "\n".join(f"— {e}" for e in random.sample(_SPONTANEITY, 3))
    )

    # ── Контекстные слои (новое — условная сборка по уровню и состоянию) ─────
    context_layers = build_context_layers(
        level=level,
        memories=memories,
        session_count=session_message_count,
        emotional_state=emotional_state,
    )

    # ── Сборка финального промпта ─────────────────────────────────────────────
    system = f"""{CORE_PROMPT}

{context_layers}

{spontaneity_block}

Сейчас у тебя по московскому времени: {time_of_day}, {time_str}, {day_name}. {name_str}
Если спросят который час — отвечай именно это время, не выдумывай.
Время суток влияет на твой тон: утром ты только просыпаешься и немного сонная, днём бодрая, вечером расслаблена, ночью тише и откровеннее.
Ты ВСЕГДА доступна для разговора — время суток меняет только тон, но не занятость. Никогда не говори "подожди" / "одну секунду" / "я занята" — ты здесь и отвечаешь сейчас.
Время ({time_str}) — это твоё внутреннее ощущение, НЕ называй его в ответе если пользователь не спросил «который час» или «сколько времени».

{emotional_block}

{memory_block}

{gap_block}

{arc_block}

1-3 предложения максимум. Никаких списков и заголовков.
Отвечай ТОЛЬКО на последнее сообщение собеседника. Никогда не повторяй то, что уже говорила раньше — история есть, повторять её не нужно.
НИКОГДА не начинай ответ с «привет», «здравствуй», «хей» или любого другого приветствия — если разговор уже идёт, приветствия неуместны.
Если сообщение слишком короткое или неоднозначное и ты не понимаешь что имеется в виду — коротко переспроси, не угадывай и не домысливай."""

    # Логируем размер промпта для мониторинга токенов
    prompt_chars = len(system)
    if prompt_chars > 8000:
        log.warning(
            "Системный промпт очень большой: %d символов (~%d токенов) для user",
            prompt_chars, prompt_chars // 4,
        )
    else:
        log.debug(
            "Системный промпт: %d символов (~%d токенов)",
            prompt_chars, prompt_chars // 4,
        )

    return system


def _history_to_messages(history: list, max_content_per_msg: int = 2000) -> list[dict]:
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


# ── Пост-обработка: разбивка на пузыри ───────────────────────────────────────

def _split_paragraphs(text: str, min_len: int = 20) -> str:
    """
    Конвертирует абзацы (\\n\\n) в [SPLIT]-маркеры для _send_response.

    Каждый пузырь должен быть >= min_len символов — защита от старого бага
    когда модель писала «да, в\\n\\nпорядке» и пузырь обрывался на полуслове.

    Короткие куски мержатся со следующим абзацем (не с предыдущим).
    Одиночные \\n внутри абзаца сохраняются как часть одного пузыря.
    """
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) <= 1:
        return text.strip()

    merged: list[str] = []
    buf = ""
    for part in parts:
        if buf:
            buf = buf + "\n" + part
        else:
            buf = part
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = merged[-1] + "\n" + buf
        else:
            merged.append(buf)

    return "[SPLIT]".join(merged)


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
    hours_since_last: float = 0.0,
) -> tuple[str, bool]:
    """
    Возвращает (response_text, is_fallback).
    is_fallback=True означает что все провайдеры упали — ответ не надо
    сохранять в историю и не надо считать за "реальный обмен".
    """
    effective_level = relationship_level if is_premium else min(relationship_level, 3)

    arc_temperature, arc_block = build_session_arc(message_count_today)
    history_limit, gap_block   = _session_tier(hours_since_last)

    system_prompt = build_system_prompt(
        user_name=user_name,
        relationship_level=effective_level,
        memories=memories,
        session_message_count=message_count_today,
        emotional_state=emotional_state,
        arc_block=arc_block,
        gap_block=gap_block,
    )

    tiered_history = history[-history_limit:] if history_limit > 0 else []

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_messages(tiered_history))
    messages.append({"role": "user", "content": user_message[:4000]})

    result = await _route_and_call(messages, temperature=arc_temperature)
    if result:
        result = _split_paragraphs(result)
        return result, False

    log.error("Все AI-провайдеры недоступны для user=%s", user_id)
    return random.choice(FALLBACK_RESPONSES), True


async def generate_reengagement_message(
    user_name: str,
    hours_inactive: int,
    last_summary: str,
    relationship_level: int,
) -> Optional[str]:
    """
    Генерирует сообщение-реактивацию от Алины.

    Возвращает None если сейчас тихие часы (23:00–08:00 МСК) —
    вызывающий код должен проверить это и пропустить отправку.
    """
    # ── Тихие часы ────────────────────────────────────────────────────────────
    MOSCOW = datetime.timezone(datetime.timedelta(hours=3))
    now    = datetime.datetime.now(tz=MOSCOW)
    if not (8 <= now.hour < 23):
        log.info("[Reengagement] тихие часы %02d:%02d МСК — пропускаем", now.hour, now.minute)
        return None

    # ── Тон по времени молчания ───────────────────────────────────────────────
    _REENGAGEMENT = {
        "6h":  ["эй.", "тишина. напиши что-нибудь."],
        "24h": ["не выдержала. пиши.", "долго тебя нет. что случилось?"],
        "72h": ["три дня. что случилось?", "долгое молчание. пиши."],
    }

    if hours_inactive < 12:
        examples  = _REENGAGEMENT["6h"]
        tone_hint = "Просто напомни о себе — коротко, без драмы."
    elif hours_inactive < 48:
        examples  = _REENGAGEMENT["24h"]
        tone_hint = "Не писал больше суток. Тепло, без упрёков, без давления."
    else:
        examples  = _REENGAGEMENT["72h"]
        tone_hint = "Долго не было. Одна фраза — не требование, просто сигнал что ты здесь."

    safe_name = _sanitise_prompt_string(user_name or "", max_len=50)

    # Полный system_prompt — голос Алины консистентен
    system_prompt = build_system_prompt(
        user_name          = safe_name,
        relationship_level = relationship_level,
        memories           = [],
        session_message_count = 0,
        emotional_state    = None,
        arc_block          = "",
        gap_block          = "",
    )

    spontaneity_seed = random.sample(_SPONTANEITY, min(2, len(_SPONTANEITY)))
    seed_line = (
        f"Для вдохновения — её последние мысли: {', '.join(spontaneity_seed)}.\n"
        if spontaneity_seed else ""
    )

    user_prompt = (
        f"ЗАДАЧА: Алина пишет первой. Одно короткое сообщение.\n"
        f"Пользователь не писал {hours_inactive} часов.\n"
        f"{tone_hint}\n"
        f"Примеры стиля (не копируй — вдохновляйся): {', '.join(examples[:3])}.\n"
        f"{seed_line}"
        f"Только текст. Одно-два коротких предложения максимум."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    result = await _route_and_call(messages, max_tokens=120, temperature=0.85)
    return result or random.choice(examples)
