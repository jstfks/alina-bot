import os
import json
import aiohttp
from database import save_memory

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_FALLBACK_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "deepseek/deepseek-r1:free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free",
    "openrouter/auto",
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
