import os
import json
import aiohttp
from database import save_memory, save_emotional_state, get_emotional_state

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_FALLBACK_MODELS = [
    "deepseek/deepseek-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


async def _extract_via_groq(prompt: str) -> dict | None:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
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
                    print(f"Memory Groq error: {data.get('error', {}).get('message', '?')}")
                    return None
                text = data["choices"][0]["message"]["content"].strip()
                text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)
    except Exception as e:
        print(f"Memory Groq exception: {e}")
        return None


async def _extract_via_openrouter(prompt: str) -> dict | None:
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
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.1,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=25)
                ) as resp:
                    data = await resp.json()
                    if "choices" not in data:
                        code = data.get("error", {}).get("code", "?")
                        print(f"Memory OpenRouter {model} failed (code {code})")
                        await asyncio.sleep(0.3)
                        continue
                    text = data["choices"][0]["message"]["content"].strip()
                    text = text.replace("```json", "").replace("```", "").strip()
                    return json.loads(text)
        except Exception as e:
            print(f"Memory OpenRouter {model} exception: {e}")
            await asyncio.sleep(0.3)
    return None


async def extract_memories(user_id: int, conversation: list[dict]):
    """Извлекаем факты: сначала Groq, резерв — OpenRouter"""
    if len(conversation) < 4:
        return

    convo_text = "\n".join([
        f"{'Пользователь' if m['role'] == 'user' else 'Алина'}: {m['content']}"
        for m in conversation[-10:]
    ])

    prompt = f"""Проанализируй разговор и извлеки факты о пользователе.

Разговор:
{convo_text}

Верни JSON объект с фактами. Только явно сказанное. Не придумывай.
Возможные ключи: name, job, city, age, hobby, pet, mood, relationship_status

Пример: {{"name": "Дима", "job": "программист"}}
Если ничего нового — верни: {{}}
Только JSON, без пояснений."""

    # Сначала Groq
    facts = await _extract_via_groq(prompt)

    # Резерв — OpenRouter
    if facts is None:
        print("Memory: Groq недоступен, переключаемся на OpenRouter...")
        facts = await _extract_via_openrouter(prompt)

    if facts:
        for key, value in facts.items():
            await save_memory(user_id, key, str(value))


def build_memory_prompt(memories: list) -> str:
    if not memories:
        return ""

    facts = {m.key: m.value for m in memories}
    lines = ["Что ты знаешь о собеседнике:"]

    if "name" in facts:
        lines.append(f"- Его зовут {facts['name']}")
    if "job" in facts:
        lines.append(f"- Работает: {facts['job']}")
    if "city" in facts:
        lines.append(f"- Живёт в: {facts['city']}")
    if "hobby" in facts:
        lines.append(f"- Увлечения: {facts['hobby']}")
    if "pet" in facts:
        lines.append(f"- Питомец: {facts['pet']}")

    skip = {"name", "job", "city", "hobby", "pet"}
    for key, value in facts.items():
        if key not in skip:
            lines.append(f"- {key}: {value}")

    lines.append("\nИспользуй эти знания естественно. Просто знай это.")
    return "\n".join(lines)


async def extract_emotional_state(user_id: int, conversation: list[dict]):
    """
    После каждой сессии анализируем эмоциональный итог разговора.
    Сохраняем: настроение, последний момент, незакрытые темы.
    """
    if len(conversation) < 4:
        return

    convo_text = "\n".join([
        f"{'Пользователь' if m['role'] == 'user' else 'Алина'}: {m['content']}"
        for m in conversation[-16:]
    ])

    prompt = f"""Проанализируй конец разговора между пользователем и девушкой Алиной.

Разговор:
{convo_text}

Верни JSON с тремя полями:
1. "mood" — каким был тон в конце разговора: "warm" (тепло, близость), "neutral" (обычно), "cold" (сухо, дистанция), "conflict" (ссора, обида), "flirty" (флирт, романтика)
2. "last_moment" — одна фраза: самый важный эмоциональный момент разговора (макс 100 символов). Что было важным? Что она сказала или он признался?
3. "open_topics" — незакрытые темы о которых начали говорить но не закончили (макс 80 символов). Если таких нет — пустая строка.

Только JSON, без пояснений. Пример:
{{"mood": "warm", "last_moment": "он рассказал про расставание, она поддержала", "open_topics": "он хотел рассказать про работу"}}"""

    # Используем Groq — быстро и бесплатно
    result = await _extract_via_groq(prompt)
    if result is None:
        result = await _extract_via_openrouter(prompt)
    if not result:
        return

    await save_emotional_state(
        user_id=user_id,
        mood_after_last_session=result.get("mood", "neutral"),
        last_emotional_moment=result.get("last_moment", ""),
        open_topics=result.get("open_topics", ""),
    )


async def update_hours_since_message(user_id: int, hours: float):
    """Обновляем сколько часов прошло с последнего сообщения."""
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


def build_emotional_state_prompt(state) -> str:
    """
    Превращает эмоциональное состояние в блок для системного промпта.
    Алина читает это и знает с каким настроением начинать.
    """
    if not state:
        return ""

    lines = ["━━━ ЭМОЦИОНАЛЬНАЯ ПАМЯТЬ — КАК ПРОШЁЛ ПРОШЛЫЙ РАЗГОВОР ━━━"]

    hours = state.hours_since_last_message
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
    mood_map = {
        "warm":     "последний разговор был тёплым — вы сблизились. начни немного теплее обычного.",
        "flirty":   "последний разговор был с флиртом и напряжением. можно продолжать в том же духе.",
        "neutral":  "последний разговор был обычным. начни как обычно.",
        "cold":     "последний разговор закончился сухо или с дистанцией. будь чуть сдержаннее в начале.",
        "conflict": "в прошлый раз была напряжённость или обида. будь немного холоднее — не сразу открывайся.",
    }
    lines.append(mood_map.get(mood, "начни как обычно."))

    if state.last_emotional_moment:
        lines.append(f"Важный момент прошлого разговора: {state.last_emotional_moment}")
        lines.append("Можешь вернуться к этому — естественно, без «помнишь ты говорил».")

    if state.open_topics:
        lines.append(f"Незакрытая тема: {state.open_topics}")
        lines.append("Если разговор зайдёт рядом — можешь сама вернуться к ней.")

    lines.append("")
    return "\n".join(lines)
