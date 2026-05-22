import os
import random
import aiohttp
from persona import ALINA
from memory import build_memory_prompt

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Список моделей по приоритету — переключается автоматически при ошибке
MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


async def _call_model(model: str, messages: list, max_tokens: int, temperature: float) -> str | None:
    """Один запрос к конкретной модели. Возвращает текст или None."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alina-bot",
        "X-Title": "Alina Bot"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.4,
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
                    error_code = data.get("error", {}).get("code", "unknown")
                    print(f"Model {model} failed (code {error_code})")
                    return None

                return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Model {model} exception: {e}")
        return None


async def _call_openrouter(messages: list, max_tokens: int = 250, temperature: float = 0.82) -> str | None:
    """Перебирает модели по списку пока одна не ответит."""
    for model in MODELS:
        result = await _call_model(model, messages, max_tokens, temperature)
        if result:
            if model != MODELS[0]:
                print(f"Fallback success: используется {model}")
            return result
        # Небольшая пауза перед следующей попыткой
        import asyncio
        await asyncio.sleep(0.5)

    print("Все модели недоступны")
    return None


def build_system_prompt(user_name: str, relationship_level: int, memories: list) -> str:
    persona = ALINA
    rel_description = persona["relationship_levels"].get(relationship_level, persona["relationship_levels"][1])
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

    result = await _call_openrouter(messages)
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
    result = await _call_openrouter(messages, max_tokens=80, temperature=0.9)
    return result or random.choice(examples)
