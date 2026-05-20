import os
import json
import aiohttp
from database import get_memories, save_memory

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-v4-flash:free"


async def extract_memories(user_id: int, conversation: list[dict]):
    """После сессии извлекаем факты о пользователе"""
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

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.1,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                data = await resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                text = text.replace("```json", "").replace("```", "").strip()
                facts = json.loads(text)
                for key, value in facts.items():
                    await save_memory(user_id, key, str(value))
    except Exception as e:
        print(f"Memory extraction error: {e}")


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
