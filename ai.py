import os
import random
import aiohttp
from persona import ALINA
from memory import build_memory_prompt

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY")

# Модели по уровням отношений
GROQ_MODEL    = "llama-3.3-70b-versatile"       # уровни 1-3
TOGETHER_MODEL = "meta-llama/Llama-3-70b-chat-hf" # уровни 4-5
FALLBACK_MODEL = "deepseek/deepseek-v4-flash:free" # резерв


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
    max_tokens: int = 250,
    temperature: float = 0.82
) -> str | None:
    """
    Умная маршрутизация по уровню отношений:
    Уровни 1-3 → Groq (быстро, бесплатно)
    Уровни 4-5 → Together AI (мягкая модерация)
    Fallback   → OpenRouter
    """
    import asyncio

    if relationship_level <= 3:
        # Основной — Groq
        result = await _call_groq(messages, max_tokens, temperature)
        if result:
            return result
        print("Groq недоступен, переключаемся на Together...")
        # Резерв — Together
        result = await _call_together(messages, max_tokens, temperature)
        if result:
            return result
    else:
        # Основной — Together AI
        result = await _call_together(messages, max_tokens, temperature)
        if result:
            return result
        print("Together недоступен, переключаемся на Groq...")
        # Резерв — Groq
        result = await _call_groq(messages, max_tokens, temperature)
        if result:
            return result

    # Последний резерв — OpenRouter
    print("Переключаемся на OpenRouter fallback...")
    await asyncio.sleep(0.5)
    return await _call_openrouter_fallback(messages, max_tokens, temperature)


def build_system_prompt(user_name: str, relationship_level: int, memories: list) -> str:
    persona = ALINA
    rel_description = persona["relationship_levels"].get(
        relationship_level, persona["relationship_levels"][1]
    )
    memory_block = build_memory_prompt(memories)

    import datetime
    hour = datetime.datetime.now().hour
    if 6 <= hour < 11:
        emotional_state = "утренняя, немного сонная"
    elif 11 <= hour < 17:
        emotional_state = "активная, в хорошем настроении"
    elif 17 <= hour < 22:
        emotional_state = "расслабленная, вечерняя"
    else:
        emotional_state = "немного усталая, но рада поговорить"

    system = f"""{persona['core_identity'].format(user_name=user_name or 'ты')}

{persona['personality']}

{persona['dialogue_rules']}

Текущий уровень отношений ({relationship_level}/5):
{rel_description}

Твоё настроение сейчас: {emotional_state}

{memory_block}

Помни: ты Алина. Живой человек. Пиши коротко и естественно."""

    return system


async def get_ai_response(
    user_id: int,
    user_message: str,
    history: list,
    user_name: str,
    relationship_level: int,
    memories: list,
    message_count_today: int
) -> str:
    system_prompt = build_system_prompt(user_name, relationship_level, memories)

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-15:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    result = await _route_and_call(messages, relationship_level)
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
