import os
import asyncio
import random
import aiohttp
from persona import ALINA
from memory import build_memory_prompt

# ── API ключи ─────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Модели ────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-2.5-flash"          # основная
DEEPSEEK_MODEL = "deepseek-chat"             # резерв 1  (V3.2)
GROQ_MODEL     = "moonshotai/kimi-k2-instruct"  # резерв 2

# OpenRouter — последний рубеж, бесплатные модели по убыванию качества
OPENROUTER_FALLBACK_MODELS = [
    "z-ai/glm-4.5-air:free",                       # хорош для диалога/roleplay
    "deepseek/deepseek-v3-0324:free",               # DeepSeek через OR
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free",
    "openrouter/auto",
]


# ══════════════════════════════════════════════════════════
# ВЫЗОВЫ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════

async def _call_gemini(messages: list, max_tokens: int, temperature: float) -> str | None:
    """Gemini 2.5 Flash — основная модель (1500 req/day бесплатно)"""
    if not GEMINI_API_KEY:
        return None

    system_prompt = ""
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            gemini_messages.append({"role": "user",  "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})

    # Gemini требует чередования user/model — убираем дубли одной роли подряд
    deduped = []
    for m in gemini_messages:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["parts"][0]["text"] += "\n" + m["parts"][0]["text"]
        else:
            deduped.append(m)

    # Первое сообщение должно быть от user
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
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                data = await resp.json()
                if "candidates" not in data:
                    print(f"[Gemini] error: {data.get('error', {}).get('message', data)}")
                    return None
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[Gemini] exception: {e}")
        return None


async def _call_deepseek(messages: list, max_tokens: int, temperature: float) -> str | None:
    """DeepSeek V3.2 — резерв 1 (5M бесплатных токенов при регистрации, потом очень дёшево)"""
    if not DEEPSEEK_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    print(f"[DeepSeek] error: {data.get('error', {}).get('message', data)}")
                    return None
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[DeepSeek] exception: {e}")
        return None


async def _call_groq(messages: list, max_tokens: int, temperature: float) -> str | None:
    """Kimi K2 через Groq — резерв 2 (быстрый, 300K tokens/day бесплатно)"""
    if not GROQ_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    print(f"[Groq/Kimi] error: {data.get('error', {}).get('message', data)}")
                    return None
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq/Kimi] exception: {e}")
        return None


async def _call_openrouter(messages: list, max_tokens: int, temperature: float) -> str | None:
    """OpenRouter — последний рубеж, перебирает бесплатные модели по очереди"""
    if not OPENROUTER_API_KEY:
        return None

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
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=35),
                ) as resp:
                    data = await resp.json()
                    if "choices" not in data:
                        code = data.get("error", {}).get("code", "?")
                        print(f"[OpenRouter] {model} failed (code {code})")
                        await asyncio.sleep(0.3)
                        continue
                    print(f"[OpenRouter] success: {model}")
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[OpenRouter] {model} exception: {e}")
            await asyncio.sleep(0.3)

    print("[OpenRouter] все модели недоступны")
    return None


# ══════════════════════════════════════════════════════════
# МАРШРУТИЗАЦИЯ
# ══════════════════════════════════════════════════════════

async def _route_and_call(
    messages: list,
    max_tokens: int = 250,
    temperature: float = 0.92,
) -> str | None:
    """
    Цепочка вызовов (единая для всех уровней и статусов):
    1. Gemini 2.5 Flash     — основная
    2. DeepSeek V3.2        — резерв 1
    3. Kimi K2 (Groq)       — резерв 2
    4. OpenRouter (GLM/DS)  — последний рубеж
    """
    # 1. Gemini
    result = await _call_gemini(messages, max_tokens, temperature)
    if result:
        return result
    print("[route] Gemini недоступен → DeepSeek")

    # 2. DeepSeek
    result = await _call_deepseek(messages, max_tokens, temperature)
    if result:
        return result
    print("[route] DeepSeek недоступен → Kimi K2 (Groq)")

    # 3. Kimi K2 через Groq
    result = await _call_groq(messages, max_tokens, temperature)
    if result:
        return result
    print("[route] Kimi K2 недоступен → OpenRouter")

    # 4. OpenRouter
    await asyncio.sleep(0.5)
    return await _call_openrouter(messages, max_tokens, temperature)


# ══════════════════════════════════════════════════════════
# СИСТЕМНЫЙ ПРОМПТ
# ══════════════════════════════════════════════════════════

def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    user_timezone_offset: int = 3,
) -> str:
    import datetime

    persona = ALINA
    rel_description = persona["relationship_levels"].get(
        relationship_level, persona["relationship_levels"][1]
    )
    memory_block = build_memory_prompt(memories)

    utc_now    = datetime.datetime.utcnow()
    local_hour = (utc_now.hour + user_timezone_offset) % 24
    local_min  = utc_now.minute
    local_date = utc_now + datetime.timedelta(hours=user_timezone_offset)
    weekday_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    day_name   = weekday_ru[local_date.weekday()]
    time_str   = f"{local_hour:02d}:{local_min:02d}"

    if 6 <= local_hour < 12:
        time_ctx = f"утро, {time_str}, {day_name}"
    elif 12 <= local_hour < 18:
        time_ctx = f"день, {time_str}, {day_name}"
    elif 18 <= local_hour < 23:
        time_ctx = f"вечер, {time_str}, {day_name}"
    else:
        time_ctx = f"ночь, {time_str}, {day_name}"

    name_str = f"Его зовут {user_name}." if user_name else ""

    system = f"""{persona['core_identity']}

{persona['personality']}

{persona['dialogue_rules']}

{rel_description}

Сейчас у него: {time_ctx}. {name_str}
Не упоминай время и день недели без причины — только если уместно.

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


# ══════════════════════════════════════════════════════════
# ПУБЛИЧНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════

async def get_ai_response(
    user_id: int,
    user_message: str,
    history: list,
    user_name: str,
    relationship_level: int,
    memories: list,
    message_count_today: int,
    is_premium: bool = False,
) -> str:
    # Бесплатные пользователи не получают уровни 4-5
    effective_level = relationship_level if is_premium else min(relationship_level, 3)

    system_prompt = build_system_prompt(user_name, effective_level, memories)

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-30:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    result = await _route_and_call(messages)
    if result:
        return result

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

    prompt = (
        f"Ты — Алина. Напиши одно короткое сообщение пользователю {user_name or ''}. "
        f"Он не писал {hours_inactive} часов. "
        f"Примеры стиля (не копируй, придумай своё): {', '.join(examples)}. "
        f"Уровень отношений: {relationship_level}/5. "
        f"Только текст, одно предложение."
    )

    messages = [{"role": "user", "content": prompt}]
    result = await _route_and_call(messages, max_tokens=80, temperature=0.9)
    return result or random.choice(examples)
