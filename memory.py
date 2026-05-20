import json
import os
from openai import AsyncOpenAI
from database import get_memories, save_memory

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


async def extract_memories(user_id: int, conversation: list[dict]):
    """После сессии извлекаем факты о пользователе"""
    if len(conversation) < 4:
        return  # слишком мало для извлечения

    convo_text = "\n".join([
        f"{'Пользователь' if m['role'] == 'user' else 'Алина'}: {m['content']}"
        for m in conversation[-10:]  # последние 10 сообщений
    ])

    prompt = f"""Проанализируй этот разговор и извлеки факты о пользователе.

Разговор:
{convo_text}

Верни JSON объект с фактами. Только то, что явно сказано. Не придумывай.
Возможные ключи: name, job, city, age, hobby, pet, mood, relationship_status, favorite_music, favorite_food

Пример ответа:
{{"name": "Дима", "job": "программист", "hobby": "играет в футбол"}}

Если ничего нового — верни пустой объект: {{}}
Верни ТОЛЬКО JSON, без пояснений."""

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        # Убираем markdown если есть
        text = text.replace("```json", "").replace("```", "").strip()
        facts = json.loads(text)

        for key, value in facts.items():
            await save_memory(user_id, key, str(value))

    except Exception as e:
        print(f"Memory extraction error: {e}")


def build_memory_prompt(memories: list) -> str:
    """Формируем блок памяти для промпта"""
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
        lines.append(f"- Есть питомец: {facts['pet']}")

    # Остальные факты
    skip = {"name", "job", "city", "hobby", "pet"}
    for key, value in facts.items():
        if key not in skip:
            lines.append(f"- {key}: {value}")

    lines.append("\nИспользуй эти знания естественно. Не говори 'я помню что ты сказал'. Просто знай это.")
    return "\n".join(lines)
