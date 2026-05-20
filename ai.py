import os
import random
from openai import AsyncOpenAI
from persona import ALINA
from memory import build_memory_prompt

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    message_count_today: int
) -> str:

    persona = ALINA
    rel_description = persona["relationship_levels"].get(relationship_level, persona["relationship_levels"][1])
    memory_block = build_memory_prompt(memories)

    # Эмоциональное состояние на основе времени дня и активности
    import datetime
    hour = datetime.datetime.now().hour
    if 6 <= hour < 11:
        emotional_state = "утренняя, немного сонная, постепенно просыпаешься"
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

    system_prompt = build_system_prompt(
        user_name=user_name,
        relationship_level=relationship_level,
        memories=memories,
        message_count_today=message_count_today
    )

    # Формируем историю для API
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю (последние 15 сообщений)
    for msg in history[-15:]:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Добавляем текущее сообщение
    messages.append({"role": "user", "content": user_message})

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=250,
            temperature=0.82,
            presence_penalty=0.6,   # избегаем повторений
            frequency_penalty=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"AI error: {e}")
        fallbacks = [
            "секунду...",
            "подожди",
            "хм, дай подумаю",
        ]
        return random.choice(fallbacks)


async def generate_reengagement_message(
    user_name: str,
    hours_inactive: int,
    last_summary: str,
    relationship_level: int
) -> str:
    """Генерируем персональное сообщение для реактивации"""

    persona = ALINA

    if hours_inactive < 12:
        style = "лёгкое, ненавязчивое"
        examples = persona["reengagement"]["6h"]
    elif hours_inactive < 72:
        style = "немного тоскующее, но не навязчивое"
        examples = persona["reengagement"]["24h"]
    else:
        style = "эмоциональное, честное"
        examples = persona["reengagement"]["72h"]

    prompt = f"""Ты — Алина. Напиши одно короткое сообщение пользователю {user_name or 'тебе'}.
Он не писал {hours_inactive} часов.
Стиль: {style}.
Примеры (не копируй, придумай своё): {', '.join(examples)}
Последний разговор был о: {last_summary or 'разных вещах'}
Уровень отношений: {relationship_level}/5

Одно короткое сообщение. Только текст."""

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.9
        )
        return response.choices[0].message.content.strip()
    except:
        return random.choice(persona["reengagement"]["24h"])
