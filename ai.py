import os
import random
import aiohttp
from persona import ALINA
from memory import build_memory_prompt

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")

# Модели
GEMINI_MODEL   = "gemini-2.0-flash"
GROQ_MODEL     = "llama-3.3-70b-versatile"
TOGETHER_MODEL = "meta-llama/Llama-3-70b-chat-hf"
FALLBACK_MODEL = "deepseek/deepseek-v4-flash:free"


async def _call_gemini(messages: list, max_tokens: int, temperature: float) -> str | None:
    """Gemini 2.0 Flash — основная модель"""
    # Конвертируем формат messages для Gemini
    system_prompt = ""
    gemini_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": gemini_messages,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=25)
            ) as resp:
                data = await resp.json()
                if "candidates" not in data:
                    print(f"Gemini error: {data}")
                    return None
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini exception: {e}")
        return None


async def _call_groq(messages: list, max_tokens: int, temperature: float) -> str | None:
    """Groq API — быстрый, бесплатный, для уровней 1-3"""
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
                timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    print(f"Groq error: {data.get('error', {}).get('message', data)}")
                    return None
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq exception: {e}")
        return None


async def _call_together(messages: list, max_tokens: int, temperature: float) -> str | None:
    """Together AI — для уровней 4-5, мягкая модерация"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TOGETHER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=25)
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    print(f"Together error: {data.get('error', {}).get('message', data)}")
                    return None
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Together exception: {e}")
        return None


# Все OpenRouter модели как резерв
OPENROUTER_FALLBACK_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "deepseek/deepseek-r1:free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free",
    "openrouter/auto",
]

async def _call_openrouter_fallback(messages: list, max_tokens: int, temperature: float) -> str | None:
    """OpenRouter — перебирает все резервные модели по очереди"""
    import asyncio
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alina-bot",
        "X-Title": "Alina Bot"
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
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    if "choices" not in data:
                        code = data.get("error", {}).get("code", "?")
                        print(f"OpenRouter {model} failed (code {code})")
                        await asyncio.sleep(0.3)
                        continue
                    print(f"OpenRouter fallback success: {model}")
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenRouter {model} exception: {e}")
            await asyncio.sleep(0.3)
    print("Все резервные модели недоступны")
    return None


async def _route_and_call(
    messages: list,
    relationship_level: int,
    is_premium: bool = False,
    max_tokens: int = 150,
    temperature: float = 0.92
) -> str | None:
    """
    Маршрутизация по уровню и статусу подписки:
    Бесплатно → уровни 1-3 → Groq
    Premium   → уровни 4-5 → Together AI
    Fallback  → OpenRouter
    """
    import asyncio

    # Уровни 4-5 только для Premium
    effective_level = relationship_level
    if not is_premium and relationship_level > 3:
        effective_level = 3

    # Gemini — основная модель для всех уровней
    result = await _call_gemini(messages, max_tokens, temperature)
    if result:
        return result
    print("Gemini недоступен, переключаемся на резерв...")

    if effective_level <= 3:
        # Резерв 1 — Groq
        result = await _call_groq(messages, max_tokens, temperature)
        if result:
            return result
        print("Groq недоступен, переключаемся на Together...")
        # Резерв 2 — Together
        result = await _call_together(messages, max_tokens, temperature)
        if result:
            return result
    else:
        # Premium резерв — Together AI
        result = await _call_together(messages, max_tokens, temperature)
        if result:
            return result
        print("Together недоступен, переключаемся на Groq...")
        result = await _call_groq(messages, max_tokens, temperature)
        if result:
            return result

    print("Переключаемся на OpenRouter fallback...")
    await asyncio.sleep(0.5)
    return await _call_openrouter_fallback(messages, max_tokens, temperature)


def build_system_prompt(user_name: str, relationship_level: int, memories: list, user_timezone_offset: int = 3) -> str:
    persona = ALINA
    rel_description = persona["relationship_levels"].get(
        relationship_level, persona["relationship_levels"][1]
    )
    memory_block = build_memory_prompt(memories)

    import datetime
    utc_now = datetime.datetime.utcnow()
    local_hour = (utc_now.hour + user_timezone_offset) % 24
    local_minute = utc_now.minute
    weekday_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    local_date = utc_now + datetime.timedelta(hours=user_timezone_offset)
    day_name = weekday_ru[local_date.weekday()]
    time_str = f"{local_hour:02d}:{local_minute:02d}"

    # Контекст времени — только факт, не настроение
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
Ты живёшь в том же городе что и он — реагируй на время естественно, как живой человек.
Не упоминай время и день недели без причины — только если это уместно в разговоре.

{memory_block}

ГЛАВНОЕ: отвечай коротко. 1-2 предложения. Живо, по-женски, естественно."""

    return system


async def get_ai_response(
    user_id: int,
    user_message: str,
    history: list,
    user_name: str,
    relationship_level: int,
    memories: list,
    message_count_today: int,
    is_premium: bool = False
) -> str:
    # Ограничиваем уровень для бесплатных пользователей
    effective_level = relationship_level if is_premium else min(relationship_level, 3)
    system_prompt = build_system_prompt(user_name, effective_level, memories)

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-15:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    result = await _route_and_call(messages, effective_level, is_premium=is_premium)
    if result:
        return result

    fallbacks = ["секунду...", "подожди немного", "хм, дай подумаю"]
    return random.choice(fallbacks)


async def generate_reengagement_message(
    user_name: str,
    hours_inactive: int,
    last_summary: str,
    relationship_level: int
) -> str:
    persona = ALINA

    if hours_inactive < 12:
        examples = persona["reengagement"]["6h"]
    elif hours_inactive < 72:
        examples = persona["reengagement"]["24h"]
    else:
        examples = persona["reengagement"]["72h"]

    prompt = f"""Ты — Алина. Напиши одно короткое сообщение пользователю {user_name or ''}.
Он не писал {hours_inactive} часов.
Примеры (не копируй, придумай своё): {', '.join(examples)}
Уровень отношений: {relationship_level}/5
Одно короткое сообщение. Только текст."""

    messages = [{"role": "user", "content": prompt}]
    result = await _route_and_call(messages, relationship_level, max_tokens=80, temperature=0.9)
    return result or random.choice(examples)
