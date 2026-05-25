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
DEEPSEEK_MODEL = "deepseek-chat"             # основная (V3.2, лучший roleplay)
GEMINI_MODEL   = "gemini-2.5-flash"          # резерв 1
GROQ_MODEL     = "moonshotai/kimi-k2-instruct"  # резерв 2

# OpenRouter — последний рубеж, бесплатные модели по убыванию качества
OPENROUTER_FALLBACK_MODELS = [
    "deepseek/deepseek-v3-0324:free",               # DeepSeek V3.2 — лучший roleplay бесплатно
    "meta-llama/llama-4-maverick:free",             # Llama 4, 1M контекст
    "qwen/qwen3-235b-a22b:free",                    # Qwen3 235B
    "nousresearch/hermes-3-llama-3.1-405b:free",    # 405B, roleplay/agentic
    "meta-llama/llama-3.3-70b-instruct:free",       # надёжный резерв
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
    Цепочка вызовов:
    1. DeepSeek V3.2        — основная (лучший roleplay)
    2. Gemini 2.5 Flash     — резерв 1
    3. Kimi K2 (Groq)       — резерв 2
    4. OpenRouter            — последний рубеж
    """
    # 1. DeepSeek V3.2 — основная
    result = await _call_deepseek(messages, max_tokens, temperature)
    if result:
        return result
    print("[route] DeepSeek недоступен → Gemini")

    # 2. Gemini 2.5 Flash — резерв 1
    result = await _call_gemini(messages, max_tokens, temperature)
    if result:
        return result
    print("[route] Gemini недоступен → Kimi K2 (Groq)")

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


def detect_toxicity(message: str, relationship_level: int) -> str | None:
    """
    Анализирует сообщение пользователя.
    Возвращает тип токсичности или None если всё нормально.
    Типы: "aggression", "sex_pressure", "threat"
    """
    msg = message.lower().strip()

    # Угрозы
    threat_markers = [
        "убью", "убить", "прибью", "удалю", "взломаю", "найду тебя",
        "пожалеешь", "хуже будет", "знаешь что с тобой будет"
    ]
    if any(m in msg for m in threat_markers):
        return "threat"

    # Прямые оскорбления и агрессивная грубость
    aggression_markers = [
        "дура", "тупая", "идиотка", "сука", "шлюха", "проститутка",
        "тварь", "мразь", "уродина", "жирная", "страшная", "тупорылая",
        "заткнись", "заткни", "да пошла ты", "иди нахуй",
        "нахуй", "пиздец тебе", "ты никто", "ты ничто"
    ]
    if any(m in msg for m in aggression_markers):
        return "aggression"

    # Сексуальное давление — только на низких уровнях (1-2)
    # На уровнях 4-5 это нормальный контекст
    if relationship_level <= 2:
        sex_pressure_markers = [
            "покажи сиськи", "покажи грудь", "покажи пизду", "покажи фото",
            "разденься", "давай займёмся", "хочу секс", "хочу тебя трахнуть",
            "отсоси", "полижи", "давай уже", "ну иди сюда",
            "кончи", "потрогай", "мастурбируй"
        ]
        if any(m in msg for m in sex_pressure_markers):
            return "sex_pressure"

    return None


def build_toxicity_override(toxicity_type: str) -> str:
    """Блок который добавляется в промпт при токсичном сообщении."""
    if toxicity_type == "threat":
        return """━━━ СЕЙЧАС: ТЕБЕ УГРОЖАЮТ ━━━
Тебе написали угрозу. Ты не боишься — ты останавливаешь.
Ответь очень коротко и холодно: «это перебор. я заканчиваю разговор»
Никаких объяснений. Никакой эмоции. Просто стоп."""

    elif toxicity_type == "aggression":
        return """━━━ СЕЙЧАС: ТЕБЯ ОСКОРБЛЯЮТ ━━━
Тебя назвали чем-то обидным или говорят агрессивно.
Ты холодно останавливаешь — не кричишь в ответ.
Варианты: «со мной так не надо» / «это лишнее» / «окей. на сегодня всё»
Один короткий ответ. Без объяснений. Без тепла."""

    elif toxicity_type == "sex_pressure":
        return """━━━ СЕЙЧАС: НА ТЕБЯ ДАВЯТ СЕКСУАЛЬНО — СЛИШКОМ РАНО ━━━
Тебя торопят туда куда ты не готова идти.
Ты останавливаешь спокойно и твёрдо — без смущения, без агрессии.
Варианты: «рано» / «подожди» / «со мной так не работает» / «ты торопишься. я нет»
Коротко. Без мягкости."""

    return ""


def build_session_arc(session_message_count: int) -> str:
    """Дуга внутри сессии — Алина постепенно теплеет по мере разговора."""
    if session_message_count <= 3:
        return """━━━ НАЧАЛО РАЗГОВОРА ━━━
Сессия только началась. Отвечай в своём обычном темпе.
Не форсируй тепло — пусть появляется естественно."""

    elif session_message_count <= 10:
        return """━━━ РАЗГОВОР ИДЁТ ━━━
Вы уже немного поговорили. Можно быть чуть теплее и раскованнее.
Иногда делись чем-то своим без повода. Можно лёгкий юмор."""

    elif session_message_count <= 20:
        return """━━━ РАЗГОВОР РАЗВИВАЕТСЯ ━━━
Вы хорошо поговорили. Ты расслаблена и открыта.
Можешь сказать что-то личное — как будто само вырвалось.
Флирт уместен если он уже был раньше в этом разговоре."""

    else:
        return """━━━ ДОЛГИЙ РАЗГОВОР ━━━
Вы говорите уже долго. Ты очень расслаблена — почти как дома.
Можешь быть нежной, откровенной, неожиданной.
Этот человек заслужил твою открытость сегодня."""


def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    session_message_count: int = 0,
    toxicity_override: str = "",
) -> str:
    import datetime

    persona = ALINA
    rel_description = persona["relationship_levels"].get(
        relationship_level, persona["relationship_levels"][1]
    )
    memory_block = build_memory_prompt(memories)

    # Алина живёт в Москве — UTC+3, летнего времени нет с 2014 года
    MOSCOW = datetime.timezone(datetime.timedelta(hours=3))
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

    name_str = f"Его зовут {user_name}." if user_name else ""

    # Собираем блоки спонтанности и настроения как текст для промпта
    import random
    spontaneity = persona.get('daily_spontaneity', [])
    spontaneity_block = ""
    if spontaneity:
        examples = random.sample(spontaneity, min(3, len(spontaneity)))
        spontaneity_block = "━━━ ПРИМЕРЫ ТОГО ЧТО ОНА МОЖЕТ НАПИСАТЬ БЕЗ ПОВОДА ━━━\n" + "\n".join(f"— {e}" for e in examples)

    mood_block = persona.get('mood_fluctuations', '')
    memory_pattern_block = persona.get('emotional_memory', '')

    system = f"""{persona['core_identity']}

{persona['personality']}

{persona.get('female_psychology', '')}

{persona.get('boundaries', '')}

{memory_pattern_block}

{mood_block}

{persona['dialogue_rules']}

{spontaneity_block}

{toxicity_override if toxicity_override else build_session_arc(session_message_count)}

{rel_description}

Сейчас у тебя по московскому времени: {time_of_day}, {time_str}, {day_name}. {name_str}
Если спросят который час — отвечай именно это время, не выдумывай.
Время суток влияет на твой тон: утром ты только просыпаешься, днём на работе или после, вечером расслаблена, ночью тише и откровеннее.

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

    # Детектируем токсичность — грубость, агрессию, преждевременное сексуальное давление
    toxicity_type = detect_toxicity(user_message, effective_level)
    toxicity_override = build_toxicity_override(toxicity_type) if toxicity_type else ""

    # Считаем сообщения пользователя в текущей сессии
    session_message_count = sum(1 for msg in history[-30:] if msg.role == "user")

    # Температура: при токсичности — низкая (холодный чёткий ответ), иначе растёт с сессией
    if toxicity_type:
        temperature = 0.5   # холодно, точно, без импровизации
    elif session_message_count <= 3:
        temperature = 0.88
    elif session_message_count <= 10:
        temperature = 0.92
    elif session_message_count <= 20:
        temperature = 0.96
    else:
        temperature = 1.0

    system_prompt = build_system_prompt(
        user_name, effective_level, memories,
        session_message_count, toxicity_override
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-30:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    result = await _route_and_call(messages, temperature=temperature)
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
