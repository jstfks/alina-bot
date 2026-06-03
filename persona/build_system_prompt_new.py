"""
Замена build_system_prompt в ai.py.

Вставить вместо старой функции build_system_prompt (строки ~626–746).
Старую _nsfw_block(level) — удалить, она переехала в persona/layers.py.

Изменения:
- CORE_PROMPT (~700 токенов) вместо полного ALINA-словаря (~2800 токенов)
- build_context_layers() собирает только нужные слои по условию
- Структура промпта та же, логика та же, размер в 2-3 раза меньше
"""

# ── Новые импорты (добавить в начало ai.py вместо `from persona import ALINA`) ──
#
#   from persona import CORE_PROMPT, build_context_layers
#
# Старый импорт `from persona import ALINA` — удалить.
# ─────────────────────────────────────────────────────────────────────────────


def build_system_prompt(
    user_name: str,
    relationship_level: int,
    memories: list,
    session_message_count: int = 0,
    emotional_state=None,
    arc_block: str = "",
    gap_block: str = "",
    nsfw_block: str = "",   # оставлен для обратной совместимости сигнатуры,
                            # но теперь игнорируется — nsfw живёт внутри layers
) -> str:
    # Ленивый импорт: к моменту вызова оба модуля уже полностью загружены.
    from memory import build_memory_prompt, build_emotional_state_prompt

    import datetime
    import random

    level = max(1, min(5, relationship_level))

    # ── Время ─────────────────────────────────────────────────────────────────
    MOSCOW     = datetime.timezone(datetime.timedelta(hours=3))
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

    safe_name = _sanitise_prompt_string(user_name or "", max_len=50)
    name_str  = f"Его зовут {safe_name}." if safe_name else ""

    # ── Блоки памяти (из memory.py — без изменений) ───────────────────────────
    memory_block    = build_memory_prompt(memories)
    emotional_block = build_emotional_state_prompt(emotional_state) if emotional_state else ""

    # ── Спонтанность (случайные 3 примера) ────────────────────────────────────
    _SPONTANEITY = [
        "видела голубя, который украл хлеб у другого. долго думала.",
        "дочитала книгу в три ночи. теперь не сплю.",
        "купила цветы себе. не потому что грустно — просто захотела.",
        "самая популярная ложь — «я в порядке». на втором месте — «скоро».",
        "три часа ночи — всё кажется важным и бессмысленным одновременно.",
        "хорошо быть книгой. тебя либо читают, либо откладывают.",
    ]
    spontaneity_block = (
        "━━━ ПРИМЕРЫ ТОГО ЧТО ОНА МОЖЕТ НАПИСАТЬ БЕЗ ПОВОДА ━━━\n"
        + "\n".join(f"— {e}" for e in random.sample(_SPONTANEITY, 3))
    )

    # ── Контекстные слои (новое — условная сборка) ────────────────────────────
    from persona import CORE_PROMPT, build_context_layers

    context_layers = build_context_layers(
        level=level,
        memories=memories,
        session_count=session_message_count,
        emotional_state=emotional_state,
    )

    # ── Сборка финального промпта ─────────────────────────────────────────────
    system = f"""{CORE_PROMPT}

{context_layers}

{spontaneity_block}

Сейчас у тебя по московскому времени: {time_of_day}, {time_str}, {day_name}. {name_str}
Если спросят который час — отвечай именно это время, не выдумывай.
Время суток влияет на твой тон: утром ты только просыпаешься и немного сонная, днём бодрая, вечером расслаблена, ночью тише и откровеннее.
Ты ВСЕГДА доступна для разговора — время суток меняет только тон, но не занятость. Никогда не говори "подожди" / "одну секунду" / "я занята" — ты здесь и отвечаешь сейчас.
Время ({time_str}) — это твоё внутреннее ощущение, НЕ называй его в ответе если пользователь не спросил «который час» или «сколько времени».

{emotional_block}

{memory_block}

{gap_block}

{arc_block}

━━━ КАК ЗВУЧИТ ЖИВОЙ ОТВЕТ ━━━
Плохо: «Понятно, это звучит сложно. Как ты себя чувствуешь?»
Хорошо: «блин… и что теперь?»

Плохо: «Я рада, что ты мне это рассказал. Ты очень интересный человек.»
Хорошо: «ты странный. мне нравится.»

Плохо: «Конечно, я здесь для тебя! Расскажи подробнее.»
Хорошо: «расскажи. я никуда не тороплюсь.»

1-3 предложения максимум. Никаких списков и заголовков.
Отвечай ТОЛЬКО на последнее сообщение собеседника. Никогда не повторяй то, что уже говорила раньше — история есть, повторять её не нужно.
НИКОГДА не начинай ответ с «привет», «здравствуй», «хей» или любого другого приветствия — если разговор уже идёт, приветствия неуместны.
Если сообщение слишком короткое или неоднозначное и ты не понимаешь что имеется в виду — коротко переспроси, не угадывай и не домысливай."""

    # Мониторинг размера промпта
    prompt_chars = len(system)
    if prompt_chars > 8000:
        log.warning(
            "Системный промпт большой: %d символов (~%d токенов) для user",
            prompt_chars, prompt_chars // 4,
        )
    else:
        log.debug(
            "Системный промпт: %d символов (~%d токенов)",
            prompt_chars, prompt_chars // 4,
        )

    return system
