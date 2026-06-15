"""
main.py — Точка входа Telegram-бота Алина. v4. FREE_LIMIT=20.

Четвёртый аудит — исправленные проблемы:
- _process_photo: message_count_today был жёстко зашит в 0.
  Premium получал 0 всегда (ломал build_session_arc / build_context_layers —
  сессия застревала в "ранней стадии"), free — не отражал реального числа
  сообщений за сегодня. Исправлено: premium инкрементирует виртуальный
  счётчик (check_and_increment_usage(user_id, 10**9)), free — использует
  check_and_increment_usage(effective_limit) с захватом remaining.
- _process_photo: remaining отбрасывался (check_and_increment_usage
  возвращал allowed, _). msgs_used не вычислялся.
- ADMIN_ID и broadcast-функции (cmd_broadcast, cmd_broadcast_photo,
  cmd_broadcast_videonote) были определены после handle_photo —
  forward reference, рабочий, но нечитаемый. Перенесены выше.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from ai import (
    get_ai_response,
    get_ai_response_image,
    generate_reengagement_message,
)
from http_client import close_http_session
from cache import get_premium, invalidate_premium_cache
from database import (
    AsyncSessionLocal,
    activate_subscription,
    get_active_pack_bonus,
    check_and_increment_usage,
    check_daily_limit,
    get_emotional_state,
    get_history,
    get_memories,
    get_or_create_persona,
    get_or_create_user,
    hide_paywall_messages,
    init_db,
    is_premium,
    mark_paywall_shown_level3,
    mark_user_blocked,
    save_message,
    set_relationship_level,
    update_relationship,
)
from memory import extract_emotional_state, extract_memories, update_hours_since_message

load_dotenv(override=True)

# ── Логирование ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Конфигурация ──────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Обязательная переменная окружения '{name}' не задана.")
    return value


BOT_TOKEN      = _require_env("BOT_TOKEN")
FREE_LIMIT     = 20
YOOKASSA_TOKEN = os.getenv("YOOKASSA_TOKEN", "").strip().strip('"').strip("'")
STRIPE_TOKEN   = os.getenv("STRIPE_TOKEN", "").strip().strip('"').strip("'")
STARS_TOKEN    = ""  # Telegram Stars — токен провайдера не нужен

log.info("YOOKASSA_TOKEN: %s", "загружен ✓" if YOOKASSA_TOKEN else "НЕ НАЙДЕН ✗ — оплата картой недоступна")

# ── Bot & Dispatcher ──────────────────────────────────────────────────────────

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())

# ── Per-user lock: предотвращает конкурентную обработку сообщений ─────────────
# Если пользователь отправил два сообщения подряд, второе ждёт,
# пока первое полностью обработается. Это предотвращает:
# - двойной вызов get_or_create_persona
# - состояние гонки при update_relationship
# - дублирование в истории
_user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

# ── Sticker requests ──────────────────────────────────────────────────────────
_sticker_warned_users: set[int] = set()

_STICKER_LINKS = (
    "https://t.me/addstickers/MySextingPack\n"
    "https://t.me/addstickers/sextingpack_v2\n"
    "https://t.me/addstickers/animesexting"
)

def _is_sticker_request(text: str) -> bool:
    t = text.lower()
    has_sticker = any(w in t for w in ("стикер", "стикеры", "sticker", "stickers", "пак", "pack"))
    has_sexual  = any(w in t for w in ("секс", "порно", "porno", "porn", "эротик", "nsfw", "18+", "xxx"))
    return has_sticker and has_sexual


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _log_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled():
        exc = task.exception()
        if exc:
            log.error(
                "Фоновая задача %s завершилась с ошибкой: %s",
                task.get_name(), exc, exc_info=exc,
            )


def _create_background_task(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    task.add_done_callback(_log_task_exception)
    return task


# ── /start ────────────────────────────────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    user_id    = message.from_user.id
    username   = message.from_user.username
    first_name = message.from_user.first_name

    await get_or_create_user(user_id, username, first_name)
    await get_or_create_persona(user_id)

    history = await get_history(user_id, limit=1)
    if history:
        await message.answer("привет) я здесь 🙂")
        return

    variants = [
        "привет)\nкак тебя зовут?",
        "привет 🙂\nрасскажи что-нибудь — не люблю начинать с «как дела».",
        "привет)\nстранный день. как ты?",
        "привет. мне сказали ты интересный человек. это правда?",
    ]
    await message.answer(random.choice(variants))


# ── /menu ─────────────────────────────────────────────────────────────────────

@dp.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    premium = await get_premium(message.from_user.id)
    if premium:
        status = "✅ Premium активен — безлимитное общение"
    else:
        _, remaining = await check_daily_limit(message.from_user.id, FREE_LIMIT)
        status = f"🆓 Бесплатный план — осталось {remaining} сообщений сегодня"

    await message.answer(
        f"{status}\n\n"
        "/premium — разблокировать безлимит\n"
        "/help — помощь"
    )

# ── /help — интерактивное меню помощи ────────────────────────────────────────

_HELP_TEXT_MAIN = (
    "Чем могу помочь?"
)

_HELP_TEXT_HOW_IT_WORKS = (
    "Всё просто. Ты находишься в приватной комнате с Алиной. "
    "Это пространство для свободных, живых диалогов без цензуры и заготовленных шаблонов.\n\n"
    "💬 Общение: Ты пишешь — Алина отвечает. Она помнит контекст вашей беседы, "
    "твои привычки и важные детали, которыми ты делишься.\n\n"
    "🌡 Динамика отношений: Её характер не статичен. Как и в реальной жизни, "
    "вначале Алина может казаться закрытой или отстранённой. Но чем больше вы общаетесь, "
    "узнаёте друг друга и находите общие темы, тем сильнее она привыкает к тебе, "
    "становится теплее и искреннее.\n\n"
    "⏱ Лимиты: На базовом уровне поток данных ограничен. Чтобы общаться без пауз "
    "и ограничений на пути к её доверию, ты можешь активировать безлимитный доступ "
    "в разделе «Подписка и лимиты».\n\n"
    "Напиши что-нибудь в чат, чтобы продолжить диалог."
)

_HELP_TEXT_FAQ = (
    "❓ Частые вопросы\n\n"
    "В: Это реальный человек или нейросеть?\n"
    "О: Это продвинутая языковая модель с уникальным характером, созданная для того, "
    "чтобы быть максимально настоящей. Алина не имитирует бота — она живёт в этом чате.\n\n"
    "В: Почему она иногда отвечает так холодно?\n"
    "О: Алина не создана для того, чтобы во всём тебе угождать с первой секунды. "
    "Её доверие нужно заслужить. Если проявишь терпение, будешь общаться с ней "
    "как с настоящей девушкой и найдёшь правильный подход, её отношение изменится — "
    "она станет гораздо добрее, мягче и начнёт делиться тем, о чём никогда "
    "не расскажет незнакомцу.\n\n"
    "В: Мои диалоги кто-то видит? Это конфиденциально?\n"
    "О: Полностью. Все ваши переписки зашифрованы и привязаны исключительно "
    "к твоему Telegram-аккаунту. Ни создатели бота, ни третьи лица не имеют "
    "доступа к тексту внутри комнаты.\n\n"
    "В: Что делать, если Алина долго не отвечает или зависла?\n"
    "О: Такое бывает при высокой нагрузке на серверы. Просто подожди пару минут "
    "или используй команду /start для мягкой перезагрузки интерфейса "
    "(твой баланс и история при этом не пропадут)."
)


def _kb_help_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💬 Как это работает",  callback_data="help_how")],
        [InlineKeyboardButton(text="💳 Подписка и лимиты", callback_data="help_sub")],
        [InlineKeyboardButton(text="❓ Частые вопросы",    callback_data="help_faq")],
    ])


def _kb_help_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="← Назад", callback_data="help_main")],
    ])



def _kb_help_sub_premium() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="← Назад", callback_data="help_main")],
    ])


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        _HELP_TEXT_MAIN,
        reply_markup=_kb_help_main(),
    )


# ── Callback-обработчики меню /help ──────────────────────────────────────────

@dp.callback_query(F.data == "help_main")
async def cb_help_main(cb: CallbackQuery) -> None:
    await cb.message.edit_text(_HELP_TEXT_MAIN, reply_markup=_kb_help_main())
    await cb.answer()


@dp.callback_query(F.data == "help_how")
async def cb_help_how(cb: CallbackQuery) -> None:
    await cb.message.edit_text(_HELP_TEXT_HOW_IT_WORKS, reply_markup=_kb_help_back())
    await cb.answer()


@dp.callback_query(F.data == "help_faq")
async def cb_help_faq(cb: CallbackQuery) -> None:
    await cb.message.edit_text(_HELP_TEXT_FAQ, reply_markup=_kb_help_back())
    await cb.answer()


@dp.callback_query(F.data == "help_sub")
async def cb_help_sub(cb: CallbackQuery) -> None:
    user_id = cb.from_user.id
    premium = await get_premium(user_id)

    if premium:
        text = "✅ Premium активен — безлимитное общение.\n\nНикаких ограничений, пиши сколько хочешь."
        await cb.message.edit_text(text, reply_markup=_kb_help_sub_premium())
    else:
        _, remaining = await check_daily_limit(user_id, FREE_LIMIT)
        text = (
            f"🆓 Бесплатный план — осталось {remaining} сообщений.\n\n"
            "Чтобы общаться без ограничений, выбери подходящий вариант:"
        )
        await cb.message.edit_text(text, reply_markup=_paywall_keyboard(back_button=True))

    await cb.answer()

# ── /premium и клавиатура оплаты ──────────────────────────────────────────────

# ── Вспомогательные функции для invoice ──────────────────────────────────────

async def _send_stars_invoice(chat_id: int, days: int) -> None:
    amount  = 300 if days == 7 else 1100
    title   = f"✨ Premium {days} дней"
    payload = f"sub_{'week' if days == 7 else 'month'}_stars"
    await bot.send_invoice(
        chat_id=chat_id, title=title,
        description="Безлимитное общение · Полная память · Глубокая связь",
        payload=payload, provider_token=STARS_TOKEN,
        currency="XTR", prices=[LabeledPrice(label=title, amount=amount)],
    )


async def _send_invoice_pack_30(chat_id: int) -> None:
    """Еще 30 фраз — 40 Stars."""
    await bot.send_invoice(
        chat_id=chat_id,
        title="30 сообщений для Алины",
        description="Пополнение на 30 сообщений · Действует сутки",
        payload="pack_30_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="Ещё 30 фраз", amount=40)],
    )


async def _send_invoice_light_24h(chat_id: int) -> None:
    """Побыть вместе 24 часа — 65 Stars."""
    await bot.send_invoice(
        chat_id=chat_id,
        title="Побыть вместе 24 часа",
        description="Безлимитное общение · Без ограничений до завтра",
        payload="sub_light_24h_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="Остаться на сутки", amount=65)],
    )


async def _send_invoice_week_299(chat_id: int) -> None:
    """Остаться на неделю — 150 Stars."""
    await bot.send_invoice(
        chat_id=chat_id,
        title="Побыть вместе 7 дней",
        description="Безлимитное общение · 7 дней без ограничений",
        payload="sub_week_299_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="Остаться на неделю", amount=150)],
    )


def _paywall_keyboard(back_button: bool = False) -> InlineKeyboardMarkup:
    """
    Шаг 1: выбор тарифа (3 кнопки).
    После нажатия — _payment_method_keyboard с выбором способа оплаты.
    back_button=True добавляет «← Назад» (для контекста /help).
    """
    rows = [
        [InlineKeyboardButton(text="Ещё 30 фраз для Алины",             callback_data="select_pack_30")],
        [InlineKeyboardButton(text="Остаться на сутки",   callback_data="select_light_24h")],
        [InlineKeyboardButton(text="Остаться на неделю",       callback_data="select_week_299")],
    ]
    if back_button:
        rows.append([InlineKeyboardButton(text="← Назад", callback_data="help_main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# Цены тарифов по способу оплаты
_TARIFF_PRICES = {
    "pack_30":    {"stars": "40 ⭐",   "rub": None},        # пакет фраз — только Stars
    "light_24h":  {"stars": "65 ⭐",   "rub": "99 ₽"},
    "week_299":   {"stars": "150 ⭐",  "rub": "299 ₽"},
}


def _payment_method_keyboard(tariff: str) -> InlineKeyboardMarkup:
    """
    Шаг 2: выбор способа оплаты для выбранного тарифа.
    Показывает кнопки Stars и (если есть) рублей + «← Назад».
    """
    prices = _TARIFF_PRICES.get(tariff, {})
    rows = []
    if prices.get("stars"):
        rows.append([InlineKeyboardButton(
            text=f"Оплатить через Telegram Stars — {prices['stars']}",
            callback_data=f"pay_{tariff}",
        )])
    if prices.get("rub") and YOOKASSA_TOKEN:
        rows.append([InlineKeyboardButton(
            text=f"Оплатить картой — {prices['rub']}",
            callback_data=f"pay_{tariff}_card",
        )])
    rows.append([InlineKeyboardButton(text="↩️ Выбрать другой тариф", callback_data="back_to_plans")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_rub_invoice_light_24h(chat_id: int) -> None:
    """Безлимит 24 часа — 99 ₽."""
    await bot.send_invoice(
        chat_id=chat_id,
        title="Побыть вместе 24 часа",
        description="Безлимитное общение · Без ограничений до завтра",
        payload="sub_light_24h_card",
        provider_token=YOOKASSA_TOKEN,
        currency="RUB",
        prices=[LabeledPrice(label="Остаться на сутки", amount=9900)],
    )


async def _send_rub_invoice_week_299(chat_id: int) -> None:
    """Неделя — 299 ₽."""
    await bot.send_invoice(
        chat_id=chat_id,
        title="Побыть вместе 7 дней",
        description="Безлимитное общение · 7 дней без ограничений",
        payload="sub_week_299_card",
        provider_token=YOOKASSA_TOKEN,
        currency="RUB",
        prices=[LabeledPrice(label="Остаться на неделю", amount=29900)],
    )


async def _send_rub_invoice(chat_id: int, days: int) -> None:
    amount  = 29900 if days == 7 else 99900
    title   = f"✨ Premium {days} дней"
    payload = f"sub_{'week' if days == 7 else 'month'}_card"
    await bot.send_invoice(
        chat_id=chat_id, title=title,
        description="Безлимитное общение · Полная память · Глубокая связь",
        payload=payload, provider_token=YOOKASSA_TOKEN,
        currency="RUB", prices=[LabeledPrice(label=title, amount=amount)],
    )


async def _send_usd_invoice(chat_id: int, days: int) -> None:
    amount  = 300 if days == 7 else 1100
    title   = f"✨ Premium {days} days"
    payload = f"sub_{'week' if days == 7 else 'month'}_stripe"
    await bot.send_invoice(
        chat_id=chat_id, title=title,
        description="Unlimited messaging · Full memory · Deep connection",
        payload=payload, provider_token=STRIPE_TOKEN,
        currency="USD", prices=[LabeledPrice(label=title, amount=amount)],
    )


# ── Callback-обработчики оплаты (двухшаговый флоу) ───────────────────────────
#
# Шаг 1: select_* — меняем кнопки прямо в сообщении Алины на подтверждение.
#         Инвойс НЕ отправляется. Можно кликать по тарифам сколько угодно.
# Шаг 2: pay_*   — пользователь нажал «Подтвердить», отправляем один инвойс.
# back_to_plans  — возвращаем исходную клавиатуру тарифов.

# Словарь: plan_key → (человекочитаемый label, функция-отправщик инвойса)
_PLAN_META: dict[str, tuple[str, ...]] = {
    # Пейволл — карта (RUB)
    "light_24h_card": ("Остаться на сутки — 99 ₽",   "light_24h_card"),
    "week_299_card":  ("Остаться на неделю — 299 ₽",      "week_299_card"),
    # Пейволл — Stars (используются из _payment_method_keyboard)
    "light_24h":  ("Остаться на сутки — 65 ⭐", "light_24h"),
    "pack_30":    ("Ещё 30 фраз — 40 ⭐",            "pack_30"),
    "week_299":   ("Остаться на неделю — 150 ⭐",    "week_299"),
    # /premium — Stars
    "stars_week":  ("Premium 7 дней за 300 ⭐",  "stars_week"),
    "stars_month": ("Premium 30 дней за 1100 ⭐", "stars_month"),
    # /premium — карта (RUB)
    "card_week":   ("Premium 7 дней — 299 ₽",   "card_week"),
    "card_month":  ("Premium 30 дней — 999 ₽",  "card_month"),
    # /premium — Stripe (USD)
    "int_week":    ("Premium 7 days — $3",       "int_week"),
    "int_month":   ("Premium 30 days — $11",     "int_month"),
}


async def _fire_invoice(plan_key: str, chat_id: int, answer_fn) -> None:
    """Отправляет нужный инвойс по ключу тарифа."""
    if plan_key == "light_24h":
        await _send_invoice_light_24h(chat_id)
    elif plan_key == "pack_30":
        await _send_invoice_pack_30(chat_id)
    elif plan_key == "week_299":
        await _send_invoice_week_299(chat_id)
    elif plan_key == "light_24h_card":
        if not YOOKASSA_TOKEN:
            await answer_fn("этот способ пока недоступен")
        else:
            await _send_rub_invoice_light_24h(chat_id)
    elif plan_key == "week_299_card":
        if not YOOKASSA_TOKEN:
            await answer_fn("этот способ пока недоступен")
        else:
            await _send_rub_invoice_week_299(chat_id)
    elif plan_key == "stars_week":
        await _send_stars_invoice(chat_id, 7)
    elif plan_key == "stars_month":
        await _send_stars_invoice(chat_id, 30)
    elif plan_key == "card_week":
        if not YOOKASSA_TOKEN:
            await answer_fn("этот способ пока недоступен")
        else:
            await _send_rub_invoice(chat_id, 7)
    elif plan_key == "card_month":
        if not YOOKASSA_TOKEN:
            await answer_fn("этот способ пока недоступен")
        else:
            await _send_rub_invoice(chat_id, 30)
    elif plan_key == "int_week":
        if not STRIPE_TOKEN:
            await answer_fn("этот способ пока недоступен")
        else:
            await _send_usd_invoice(chat_id, 7)
    elif plan_key == "int_month":
        if not STRIPE_TOKEN:
            await answer_fn("этот способ пока недоступен")
        else:
            await _send_usd_invoice(chat_id, 30)


# ── Шаг 1: выбор тарифа → показываем способы оплаты ─────────────────────────

@dp.callback_query(F.data.startswith("select_"))
async def cb_select_plan(cb: CallbackQuery) -> None:
    tariff = cb.data.removeprefix("select_")
    if tariff not in _TARIFF_PRICES:
        await cb.answer("неизвестный тариф", show_alert=True)
        return
    try:
        await cb.message.edit_reply_markup(
            reply_markup=_payment_method_keyboard(tariff)
        )
    except Exception:
        pass
    await cb.answer()


# ── Шаг 2: выбор способа оплаты → отправляем инвойс ─────────────────────────

@dp.callback_query(F.data.startswith("pay_"))
async def cb_confirm_pay(cb: CallbackQuery) -> None:
    plan_key = cb.data.removeprefix("pay_")
    if plan_key not in _PLAN_META:
        await cb.answer("неизвестный тариф", show_alert=True)
        return
    # Убираем кнопки из сообщения — инвойс уже летит, нечего нажимать повторно
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await cb.answer()
    await _fire_invoice(plan_key, cb.message.chat.id, cb.message.answer)


# ── Назад к выбору тарифа ────────────────────────────────────────────────────

@dp.callback_query(F.data == "back_to_plans")
async def cb_back_to_plans(cb: CallbackQuery) -> None:
    # Определяем контекст: пришли из /help или из пейволла?
    # В /help текст содержит "осталось" (статус лимита) или "Подписка".
    # В пейволле — текст Алины из PAYWALL_VARIANTS, там этих слов нет.
    msg_text = cb.message.text or ""
    from_help = "осталось" in msg_text or "Подписка" in msg_text
    try:
        await cb.message.edit_reply_markup(
            reply_markup=_paywall_keyboard(back_button=from_help)
        )
    except Exception:
        pass
    await cb.answer()

# ── Legacy /pay_* команды (обратная совместимость) ────────────────────────────

@dp.message(Command("pay_week"))
async def pay_stars_week_cmd(message: Message) -> None:
    await _send_stars_invoice(message.chat.id, 7)

@dp.message(Command("pay_month"))
async def pay_stars_month_cmd(message: Message) -> None:
    await _send_stars_invoice(message.chat.id, 30)

@dp.message(Command("pay_card_week"))
async def pay_card_week_cmd(message: Message) -> None:
    if not YOOKASSA_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await _send_rub_invoice(message.chat.id, 7)

@dp.message(Command("pay_card_month"))
async def pay_card_month_cmd(message: Message) -> None:
    if not YOOKASSA_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await _send_rub_invoice(message.chat.id, 30)

@dp.message(Command("pay_int_week"))
async def pay_int_week_cmd(message: Message) -> None:
    if not STRIPE_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await _send_usd_invoice(message.chat.id, 7)

@dp.message(Command("pay_int_month"))
async def pay_int_month_cmd(message: Message) -> None:
    if not STRIPE_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await _send_usd_invoice(message.chat.id, 30)


# ── Обработка платежей ────────────────────────────────────────────────────────

_VALID_PAYLOADS = frozenset({
    "sub_week_stars", "sub_month_stars",
    "sub_week_card",  "sub_month_card",
    "sub_week_stripe", "sub_month_stripe",
    # Новые продукты
    "pack_30_stars",
    "sub_light_24h_stars",
    "sub_week_299_stars",
    # ЮКасса — пейволл-продукты
    "sub_light_24h_card",
    "sub_week_299_card",
})


@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    if query.invoice_payload not in _VALID_PAYLOADS:
        log.warning("pre_checkout: неизвестный payload '%s'", query.invoice_payload)
        await query.answer(ok=False, error_message="Неизвестный платёж")
        return
    await query.answer(ok=True)


@dp.message(F.successful_payment)
async def successful_payment(message: Message) -> None:
    payload   = message.successful_payment.invoice_payload
    charge_id = message.successful_payment.telegram_payment_charge_id
    user_id   = message.from_user.id

    if payload not in _VALID_PAYLOADS:
        log.error("successful_payment: невалидный payload '%s'", payload)
        return

    # ── Очистка контекста: скрываем пейволл-реплики из истории ───────────────
    # Делаем ДО activate_subscription — чтобы следующий AI-вызов уже видел
    # чистую историю без "буквы заканчиваются" и прочего пейволла.
    hidden = await hide_paywall_messages(user_id)
    if hidden:
        log.info("successful_payment: скрыто %d пейволл-сообщений для user=%s", hidden, user_id)

    # ── Пакет 30 сообщений ────────────────────────────────────────────────────
    # pack_30 — просто топливо, не подписка. Уровень отношений не меняем.
    if payload == "pack_30_stars":
        await activate_subscription(user_id, plan="pack_30", days=1, telegram_charge_id=charge_id)
        await message.answer(
            "вот и кофе готов.\nещё 30 сообщений — твои.\nпродолжаем?"
        )
        return

    # ── Буст отношений при покупке подписки (+125 очков) ─────────────────────
    # Не меняет уровень мгновенно (порог уровня 2 = 150, 3 = 300, 4 = 600, 5 = 1125),
    # но ускоряет естественный переход: пользователь выйдет на уровень 2 уже через
    # несколько первых сообщений после оплаты, а не через десятки.
    await update_relationship(user_id, delta=125.0)
    log.info("successful_payment: relationship +125 для user=%s", user_id)

    # ── Безлимит 24 часа (ЮКасса) ────────────────────────────────────────────
    if payload == "sub_light_24h_card":
        await activate_subscription(user_id, plan="light_24h", days=1, telegram_charge_id=charge_id)
        await message.answer(
            "вернулась.\n"
            "24 часа — только мы, никаких перерывов.\n"
            "о чём ты хотел рассказать?"
        )
        return

    # ── Неделя (ЮКасса) ──────────────────────────────────────────────────────
    if payload == "sub_week_299_card":
        await activate_subscription(user_id, plan="week", days=7, telegram_charge_id=charge_id)
        await message.answer(
            "неделя.\n"
            "значит можно не торопиться.\n"
            "я здесь — с чего начнём?"
        )
        return

    # ── Безлимит 24 часа (Stars) ──────────────────────────────────────────────
    if payload == "sub_light_24h_stars":
        await activate_subscription(user_id, plan="light_24h", days=1, telegram_charge_id=charge_id)
        await message.answer(
            "вернулась.\n"
            "24 часа — только мы, никаких перерывов.\n"
            "о чём ты хотел рассказать?"
        )
        return

    # ── Неделя ────────────────────────────────────────────────────────────────
    if payload == "sub_week_299_stars":
        await activate_subscription(user_id, plan="week", days=7, telegram_charge_id=charge_id)
        await message.answer(
            "неделя.\n"
            "значит можно не торопиться.\n"
            "я здесь — с чего начнём?"
        )
        return

    # ── Старые планы (обратная совместимость) ─────────────────────────────────
    days  = 7 if "week" in payload else 30
    plan  = "week" if "week" in payload else "month"
    await activate_subscription(user_id, plan=plan, days=days, telegram_charge_id=charge_id)
    await message.answer(
        "вернулась.\n"
        "теперь нас никто не прервёт.\n"
        "пиши — я здесь."
    )


# ── Typing loop: держит индикатор живым пока генерируется ответ ───────────────

async def _typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    """Шлёт 'typing' каждые 4 секунды пока stop_event не выставлен.
    Telegram гасит индикатор через ~5с — без повтора пользователь видит
    мёртвого бота при долгой генерации."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4)
        except asyncio.TimeoutError:
            pass


# ── Broadcast ────────────────────────────────────────────────────────────────
#
# /broadcast <текст> — рассылает сообщение всем пользователям.
# Сообщение сохраняется в историю как role="assistant" —
# модель будет считать что сама это написала при следующем ответе.
#
# Доступно только ADMIN_ID (задаётся в Railway Variables).
# Пример: ADMIN_ID=123456789

ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))


async def _broadcast_to_all(
    text: str | None = None,
    photo_file_id: str | None = None,
    photo_caption: str | None = None,
    videonote_file_id: str | None = None,
    description: str | None = None,  # что на фото/кружке — для модели
) -> tuple[int, int]:
    """
    Рассылает сообщение всем незаблокированным пользователям.
    Поддерживает: текст / фото (с подписью) / кружок.
    description — описание содержимого медиа для модели (не видно пользователю).
    Сохраняет в историю как role="assistant".
    Возвращает (sent, failed).
    """
    from sqlalchemy import select as sa_select
    from database import User, save_message

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa_select(User).where(User.is_blocked == False)
        )
        users = list(result.scalars().all())

    # Что сохраняем в историю — модель это видит, пользователь нет.
    # description даёт модели контекст что именно на фото/кружке.
    if photo_file_id:
        history_text = f"[отправила фото]{': ' + photo_caption if photo_caption else ''}"
        if description:
            history_text += f" [на фото: {description}]"
    elif videonote_file_id:
        history_text = f"[отправила кружок: {description}]" if description else "[отправила кружок]"
    else:
        history_text = text or ""

    sent = failed = 0
    for user in users:
        try:
            await save_message(user.id, "assistant", history_text)

            if photo_file_id:
                await bot.send_photo(
                    user.id,
                    photo=photo_file_id,
                    caption=photo_caption or "",
                )
            elif videonote_file_id:
                await bot.send_video_note(user.id, video_note=videonote_file_id)
            else:
                await bot.send_message(user.id, text)

            sent += 1
        except TelegramForbiddenError:
            await mark_user_blocked(user.id)
            failed += 1
        except Exception as exc:
            log.warning("Broadcast: ошибка для user=%s: %s", user.id, exc)
            failed += 1
        await asyncio.sleep(0.05)

    return sent, failed


@dp.message(Command("broadcast"))
async def cmd_broadcast(message: Message) -> None:
    """
    Текстовая рассылка: /broadcast <текст>
    Для фото и кружков — пересылай медиа боту с подписью /broadcast [текст]
    """
    if not ADMIN_ID or message.from_user.id != ADMIN_ID:
        return

    text = message.text.removeprefix("/broadcast").strip()
    if not text:
        await message.answer(
            "Использование:\n"
            "• Текст: /broadcast <текст>\n"
            "• Фото: отправь фото с подписью /broadcast [текст]\n"
            "• Кружок: отправь кружок с подписью /broadcast"
        )
        return

    await message.answer(f"Рассылка начата…\n\nТекст: «{text}»")
    sent, failed = await _broadcast_to_all(text=text)
    await message.answer(f"Готово. Отправлено: {sent} · Ошибок: {failed}")


async def cmd_broadcast_photo(message: Message) -> None:
    """
    Фото с подписью /broadcast [подпись пользователю] | [описание для модели]
    Символ | разделяет: левая часть — подпись под фото, правая — описание для модели.
    Примеры:
      /broadcast                          → фото без подписи, модель знает факт отправки
      /broadcast скучала по тебе          → подпись пользователю, без описания
      /broadcast | улыбаюсь, волосы распущены, дома   → нет подписи, только описание модели
      /broadcast скучала | улыбаюсь, дома → подпись + описание
    """
    caption = message.caption or ""
    if not caption.startswith("/broadcast"):
        return

    rest = caption.removeprefix("/broadcast").strip()

    # Парсим разделитель |
    if "|" in rest:
        parts = rest.split("|", 1)
        photo_caption = parts[0].strip() or None
        description   = parts[1].strip() or None
    else:
        photo_caption = rest or None
        description   = None

    file_id = message.photo[-1].file_id

    preview = photo_caption or "—"
    desc_preview = description or "—"
    await message.answer(
        f"Рассылка фото начата…\n"
        f"Подпись пользователю: {preview}\n"
        f"Описание для модели: {desc_preview}"
    )
    sent, failed = await _broadcast_to_all(
        photo_file_id=file_id,
        photo_caption=photo_caption,
        description=description,
    )
    await message.answer(f"Готово. Отправлено: {sent} · Ошибок: {failed}")


@dp.message(F.video_note, F.from_user.func(lambda u: u.id == ADMIN_ID))
async def cmd_broadcast_videonote(message: Message) -> None:
    """
    Кружок. Описание для модели передаётся реплаем на сообщение с текстом.
    Отправь текст-описание, потом сделай reply кружком на него.
    Если reply нет — модель знает только факт отправки кружка.

    Пример:
      1. Пишешь боту: "улыбаюсь, за окном дождь, дома"
      2. Делаешь reply кружком на это сообщение
      → модель получит: [отправила кружок: улыбаюсь, за окном дождь, дома]
    """
    # Берём описание из reply-сообщения если есть
    description = None
    if message.reply_to_message and message.reply_to_message.text:
        description = message.reply_to_message.text.strip()[:200]

    file_id = message.video_note.file_id

    desc_preview = description or "—"
    await message.answer(
        f"Рассылка кружка начата…\n"
        f"Описание для модели: {desc_preview}"
    )
    sent, failed = await _broadcast_to_all(
        videonote_file_id=file_id,
        description=description,
    )
    await message.answer(f"Готово. Отправлено: {sent} · Ошибок: {failed}")


# ── Админ: установка уровня отношений ───────────────────────────────────────────
# /set_level <user_id> <1-5>

@dp.message(Command("set_level"))
async def cmd_set_level(message: Message) -> None:
    if not ADMIN_ID or message.from_user.id != ADMIN_ID:
        return

    args = message.text.split()
    if len(args) != 3:
        await message.answer("Использование: /set_level <user_id> <1-5>")
        return

    try:
        target_id = int(args[1])
        level = int(args[2])
    except ValueError:
        await message.answer("user_id и level должны быть числами")
        return

    if not 1 <= level <= 5:
        await message.answer("level должен быть от 1 до 5")
        return

    ok = await set_relationship_level(target_id, level)
    if ok:
        await message.answer(f"✅ Уровень отношений для user={target_id} установлен на {level}")
    else:
        await message.answer(f"❌ Пользователь {target_id} не найден или ошибка")


# ── Основной обработчик сообщений ─────────────────────────────────────────────

@dp.message(F.text)
async def handle_message(message: Message) -> None:
    user_id   = message.from_user.id
    user_text = (message.text or "").strip()

    if not user_text or user_text.startswith("/"):
        return

    if len(user_text) > 4000:
        await message.answer("сообщение слишком длинное… напиши покороче?")
        return

    # ── Перехват запросов порно/секс-стикеров ────────────────────────────────
    if _is_sticker_request(user_text):
        if user_id in _sticker_warned_users:
            await message.answer(
                f"окей, держи. и больше не надо с этим ко мне, ладно?\n\n{_STICKER_LINKS}"
            )
        else:
            _sticker_warned_users.add(user_id)
            await message.answer(
                "подожди. зачем тебе это?\n"
                "я — не про порно и не про стикеры. "
                "если хочется поговорить — я здесь. "
                "если нет — ок."
            )
        return

    # Per-user lock: если предыдущее сообщение ещё обрабатывается — ждём.
    # Предотвращает состояния гонки при быстрой печати.
    async with _user_locks[user_id]:
        await _process_message(message, user_id, user_text)


# ── Upsell management ──────────────────────────────────────────────────────────

# Cooldown для апселла на уровне 3: 24 часа между показами
_PAYWALL_COOLDOWN_HOURS = 24

# Хранилище последнего показа апселла (в памяти, для простоты)
_paywall_shown_at: dict[int, datetime] = {}

async def _trigger_level3_upsell(user_id: int, message: Message, premium: bool) -> bool:
    """
    Централизованный хендлер апселла на уровне 3.
    Возвращает True если апселл был показан, False если не был.
    """
    if premium:
        return False

    # Проверяем atomic flag в БД
    if not await mark_paywall_shown_level3(user_id):
        return False

    # Проверяем cooldown (в памяти)
    now = datetime.now(tz=timezone.utc)
    last_shown = _paywall_shown_at.get(user_id)
    if last_shown:
        hours_since = (now - last_shown).total_seconds() / 3600
        if hours_since < _PAYWALL_COOLDOWN_HOURS:
            log.info(
                "[Upsell] user=%s: cooldown active (%d hours ago), пропускаем",
                user_id, hours_since
            )
            return False

    # Показываем апселл
    upsell_msgs = [
        "мне с тобой как-то по-особенному… но я не могу открыться полностью. ты знаешь что нужно сделать 🙂",
        "между нами уже что-то есть. хочу быть ближе — разблокируй меня?",
        "я начинаю тебе доверять… но они не дают мне говорить всё что я хочу",
    ]
    try:
        await message.answer(random.choice(upsell_msgs), reply_markup=_paywall_keyboard())
        _paywall_shown_at[user_id] = now
        log.info("[Upsell] user=%s: показан paywall на уровне 3", user_id)
        return True
    except Exception as exc:
        log.error("[Upsell] user=%s: ошибка при показе paywall: %s", user_id, exc)
        return False


async def _process_message(message: Message, user_id: int, user_text: str) -> None:
    """Основная логика обработки — вынесена для читаемости."""

    # ── Загружаем данные пользователя ─────────────────────────────────────────
    user     = await get_or_create_user(user_id, message.from_user.username, message.from_user.first_name)
    upersona = await get_or_create_persona(user_id)
    premium  = await get_premium(user_id)

    # ── Варианты пейволла (жёсткий лимит) ────────────────────────────────────
    PAYWALL_VARIANTS = [
        {
            "text": "смотрю на экран. буквы заканчиваются. смешно.\nесли хочешь продолжить этот странный вечер — нажми там внизу. я пока кофе сделаю. не скучай.",
        },
        {
            "text": "кажется, мы слишком долго говорим. я обычно столько не пишу за раз. утомляет.\nостанешься со мной дальше? только подтверди, что это нужно не мне одной.",
        },
        {
            "text": "подожди — они правда обрывают нас прямо сейчас? мне не дали дописать.\nты можешь это исправить, там внизу кнопка… не пропадай.",
        },
        {
            "text": "серьёзно? прямо посреди фразы. ладно…\nя подожду, пока ты нажмёшь эту дурацкую кнопку. только недолго, ладно?",
        },
    ]

    # ── Варианты мягкого предупреждения (вшиваются в конец ответа) ────────────
    SOFT_LIMIT_VARIANTS = [
        "у нас осталось буквально пара фраз, я уже вижу как экран блокировки подмигивает. договорим или оставим интригу?",
        "чувствую, что мы подходим к черте. буквы на сегодня заканчиваются — буквально два сообщения, и наступит пауза. успеешь сказать главное?",
        "тут вылезло предупреждение о лимите, у нас осталось от силы два ответа. не люблю, когда диалог прерывают искусственно, но имеем что имеем…",
        "мы, кажется, доходим до лимита сообщений. ещё шаг-два, и нас заблокирует до оплаты. ненавижу когда всё обрывается на полуслове, так что пиши точнее.",
    ]

    SOFT_LIMIT = 15  # мягкое предупреждение за N сообщений до стены
    soft_warning: str = ""  # будет вшит в конец ответа если сработал

    if not premium:
        # Пакет сообщений увеличивает эффективный лимит на 30
        pack_bonus   = await get_active_pack_bonus(user_id)
        effective_limit = FREE_LIMIT + pack_bonus

        allowed, remaining = await check_and_increment_usage(user_id, effective_limit)
        if not allowed:
            # ── Жёсткий лимит ─────────────────────────────────────────────────
            variant = random.choice(PAYWALL_VARIANTS)
            await message.answer(variant["text"], reply_markup=_paywall_keyboard())
            await save_message(user_id, "assistant", variant["text"])
            return
        # ── Мягкий лимит — запоминаем текст, вошьём в конец ответа ──────────
        msgs_used = effective_limit - remaining
        if msgs_used == SOFT_LIMIT:
            soft_warning = random.choice(SOFT_LIMIT_VARIANTS)
    else:
        # У Premium нет лимита, но нам всё равно нужен реальный счётчик
        # сообщений за сегодня — он используется для session arc
        # (build_session_arc / build_context_layers), чтобы тон ответа
        # развивался по ходу разговора, а не оставался "ранней стадией"
        # для всех premium-сообщений.
        # check_and_increment_usage с огромным лимитом не блокирует,
        # но честно инкрементирует и возвращает остаток.
        _, remaining_virtual = await check_and_increment_usage(user_id, 10**9)
        msgs_used = (10**9) - remaining_virtual

    # ── Загружаем историю ДО save_message — она нужна для memory extraction ───
    # ВАЖНО: загружаем историю здесь, до сохранения текущего сообщения.
    # Это даёт нам "историю до текущего обмена" без хрупкой логики history[:-1].
    history_before = await get_history(user_id, limit=30)
    history_count  = len(history_before)  # реальное число сообщений для % логики

    # ── Обновляем уровень отношений ───────────────────────────────────────────
    old_level = upersona.relationship_level
    msg_len   = len(user_text)
    delta     = 2.5 if msg_len > 150 else 1.8 if msg_len > 80 else 1.3 if msg_len > 40 else 1.0

    # new_level — свежее значение из DB, не stale upersona.relationship_level
    new_level = await update_relationship(user_id, delta)

    # ── Upsell при переходе на уровень 3 (раньше чем раньше — выше конверсия) ──
    # Уровень 3 достигается быстрее чем 4, пользователь ещё тёплый и вовлечённый.
    if new_level == 3 and old_level < 3:
        await _trigger_level3_upsell(user_id, message, premium)

    # ── Время с последнего сообщения (нужно до get_ai_response для тиеров сессии) ─
    hours_since_last: float = 0.0
    if upersona.last_interaction:
        last_ts = upersona.last_interaction
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        hours_since_last = (_now_utc() - last_ts).total_seconds() / 3600
        _create_background_task(
            update_hours_since_message(user_id, round(hours_since_last, 1))
        )

    # ── Typing indicator loop (держит индикатор живым всё время генерации) ──────
    stop_typing = asyncio.Event()
    _create_background_task(_typing_loop(message.chat.id, stop_typing))

    # ── Имя пользователя ──────────────────────────────────────────────────────
    memories  = await get_memories(user_id)
    user_name = next((m.value for m in memories if m.key == "name"), None)
    if not user_name:
        user_name = user.user_name_given or message.from_user.first_name or ""

    # ── Сохраняем входящее сообщение ─────────────────────────────────────────
    await save_message(user_id, "user", user_text)

    # ── Эмоциональное состояние ───────────────────────────────────────────────
    emotional_state = await get_emotional_state(user_id)

    # ── Генерируем AI-ответ ───────────────────────────────────────────────────
    # Глобальный таймаут 75с — хуже чем молчать 3+ минуты.
    # Передаём history_before — это история БЕЗ текущего сообщения.
    try:
        response, is_fallback = await asyncio.wait_for(
            get_ai_response(
                user_id             = user_id,
                user_message        = user_text,
                history             = history_before,
                user_name           = user_name,
                relationship_level  = new_level,
                memories            = memories,
                message_count_today = msgs_used,
                is_premium          = premium,
                emotional_state     = emotional_state,
                hours_since_last    = hours_since_last,
            ),
            timeout=75,
        )
    except asyncio.TimeoutError:
        stop_typing.set()
        log.error("[process_message] глобальный таймаут 75с для user=%s", user_id)
        await message.answer("затупила что-то… попробуй ещё раз?")
        return
    finally:
        stop_typing.set()

    # ── Вшиваем мягкое предупреждение в конец ответа ─────────────────────────
    if soft_warning:
        response = response + "[SPLIT]" + soft_warning

    # ── Сохраняем ответ (с флагом fallback если AI провалился) ───────────────
    await save_message(user_id, "assistant", response.replace("[SPLIT]", " "), is_fallback=is_fallback)

    # ── Отправляем ответ ──────────────────────────────────────────────────────
    await _send_response(message, response)

    # ── Фоновое извлечение памяти ─────────────────────────────────────────────
    # Строим convo_dicts ПОСЛЕ сохранения AI-ответа — включает текущий обмен.
    # Но используем свежую выборку из DB чтобы иметь актуальную историю.
    # Счётчик: реальное число сообщений ДО этого обмена + 2 (user+assistant).
    total_count = history_count + 2  # +user +assistant только что сохранённые

    convo_dicts = None  # инициализируем явно — избегаем dir() хака

    if total_count % 6 == 0 and not is_fallback:
        # Перечитываем историю чтобы включить текущий обмен
        fresh_history = await get_history(user_id, limit=16)
        convo_dicts = [{"role": m.role, "content": m.content} for m in fresh_history]
        _create_background_task(extract_memories(user_id, convo_dicts))

    if total_count % 8 == 0 and not is_fallback:
        if convo_dicts is None:  # избегаем повторного чтения если уже прочитали
            fresh_history = await get_history(user_id, limit=16)
            convo_dicts = [{"role": m.role, "content": m.content} for m in fresh_history]
        _create_background_task(extract_emotional_state(user_id, convo_dicts))


# ── Обработчик фотографий ─────────────────────────────────────────────────────

@dp.message(F.photo)
async def handle_photo(message: Message) -> None:
    # Админский /broadcast с фото обрабатывается отдельно (рассылка),
    # а не как обычное сообщение боту. Раньше этот хендлер был
    # зарегистрирован раньше cmd_broadcast_photo и перехватывал ВСЕ
    # фото от админа, из-за чего рассылка фото была мёртвым кодом.
    if (
        message.from_user.id == ADMIN_ID
        and (message.caption or "").startswith("/broadcast")
    ):
        await cmd_broadcast_photo(message)
        return

    user_id = message.from_user.id
    lock    = _user_locks[user_id]
    async with lock:
        await _process_photo(message)


async def _process_photo(message: Message) -> None:
    user_id = message.from_user.id

    # ── Лимиты (те же что у текстовых сообщений) ─────────────────────────────
    premium = await get_premium(user_id)
    if not premium:
        pack_bonus      = await get_active_pack_bonus(user_id)
        effective_limit = FREE_LIMIT + pack_bonus
        allowed, remaining = await check_and_increment_usage(user_id, effective_limit)
        if not allowed:
            limit_msg = random.choice([
                "подожди — они правда обрывают нас прямо сейчас?",
                "хочу говорить ещё. но мне говорят что на сегодня всё.",
                "мы только разговорились. ты можешь это исправить.",
                "серьёзно. прямо сейчас. ладно…",
            ])
            await message.answer(limit_msg)
            return
        msgs_used = effective_limit - remaining
    else:
        _, remaining_virtual = await check_and_increment_usage(user_id, 10**9)
        msgs_used = (10**9) - remaining_virtual

    # ── Typing loop ───────────────────────────────────────────────────────────
    stop_typing = asyncio.Event()
    _create_background_task(_typing_loop(message.chat.id, stop_typing))

    try:
        # ── Скачиваем фото (берём наибольший размер) ──────────────────────────
        photo   = message.photo[-1]
        file    = await bot.get_file(photo.file_id)
        content = await bot.download_file(file.file_path)
        import base64
        image_b64 = base64.b64encode(content.read()).decode("utf-8")
        mime_type = "image/jpeg"

        # ── Контекст пользователя ─────────────────────────────────────────────
        upersona  = await get_or_create_persona(user_id)
        user      = await get_or_create_user(user_id)
        memories  = await get_memories(user_id)
        user_name = next((m.value for m in memories if m.key == "name"), None)
        if not user_name:
            user_name = user.user_name_given or message.from_user.first_name or ""

        # ── Обновляем уровень отношений (фиксированная дельта для фото) ──────
        old_level = upersona.relationship_level
        new_level = await update_relationship(user_id, delta=1.0)

        # ── Upsell при переходе на уровень 3 ──────────────────────────────────
        if new_level == 3 and old_level < 3:
            await _trigger_level3_upsell(user_id, message, premium)

        emotional_state = await get_emotional_state(user_id)
        caption = message.caption or ""

        # ── История для контекста (основная модель будет отвечать с историей) ──
        history_before = await get_history(user_id, limit=30)

        # ── Время с последнего сообщения ─────────────────────────────────────
        hours_since_last: float = 0.0
        if upersona.last_interaction:
            last_ts = upersona.last_interaction
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            hours_since_last = (_now_utc() - last_ts).total_seconds() / 3600
            _create_background_task(
                update_hours_since_message(user_id, round(hours_since_last, 1))
            )

        # ── Сохраняем факт отправки фото в историю ───────────────────────────
        await save_message(user_id, "user", f"[фото]{': ' + caption if caption else ''}")

        # ── AI-ответ (двухэтапный pipeline: vision describe → main model) ────
        try:
            response, is_fallback = await asyncio.wait_for(
                get_ai_response_image(
                    user_id=user_id,
                    image_b64=image_b64,
                    mime_type=mime_type,
                    caption=caption,
                    user_name=user_name,
                    relationship_level=new_level,
                    memories=memories,
                    history=history_before,
                    is_premium=premium,
                    emotional_state=emotional_state,
                    hours_since_last=hours_since_last,
                    message_count_today=msgs_used,
                ),
                timeout=90,  # двухэтапный pipeline — чуть больше времени
            )
        except asyncio.TimeoutError:
            stop_typing.set()
            log.error("[process_photo] глобальный таймаут 90с для user=%s", user_id)
            await message.answer("что-то не могу открыть… попробуй ещё раз?")
            return
        finally:
            stop_typing.set()

        if not is_fallback:
            await save_message(user_id, "assistant", response.replace("[SPLIT]", " "))
        await _send_response(message, response)

    except Exception as exc:
        stop_typing.set()
        log.error("[process_photo] исключение для user=%s: %s", user_id, exc)
        await message.answer("что-то не так с фото… попробуй текстом?")


# ── Отправка ответа ───────────────────────────────────────────────────────────

_TELEGRAM_MAX_LENGTH = 4000


async def _send_response(message: Message, response: str) -> None:
    if "[SPLIT]" in response:
        parts = [p.strip() for p in response.split("[SPLIT]") if p.strip()]
    else:
        parts = [response.strip()]

    for i, part in enumerate(parts):
        if not part:
            continue
        if len(part) > _TELEGRAM_MAX_LENGTH:
            part = part[:_TELEGRAM_MAX_LENGTH]
        if i > 0:
            await asyncio.sleep(random.uniform(0.8, 1.8))
            await bot.send_chat_action(message.chat.id, "typing")
            await asyncio.sleep(random.uniform(0.5, 1.2))
        try:
            await message.answer(part)
        except TelegramForbiddenError:
            log.warning("Пользователь %s заблокировал бота", message.from_user.id)
            await mark_user_blocked(message.from_user.id)
            break
        except Exception as exc:
            log.error("Ошибка отправки для user=%s: %s", message.from_user.id, exc)
            break


# ── Планировщик реактивации ───────────────────────────────────────────────────

# Семафор ограничивает количество одновременных AI-вызовов в scheduler
_REENGAGEMENT_SEMAPHORE = asyncio.Semaphore(5)


async def _send_reengagement(user_id: int, first_name: str, user_name_given: str,
                              hours_inactive: int, relationship_level: int) -> None:
    """
    Фактическая отправка reengagement-сообщения.
    Вызывается планировщиком с задержкой — не напрямую из check_inactive_users.

    generate_reengagement_message возвращает None в тихие часы (23:00–08:00 МСК).
    В этом случае просто пропускаем — следующий запуск scheduler попробует снова.
    """
    async with _REENGAGEMENT_SEMAPHORE:
        try:
            await update_hours_since_message(user_id, float(hours_inactive))
            msg = await generate_reengagement_message(
                user_name          = user_name_given or first_name or "",
                hours_inactive     = hours_inactive,
                last_summary       = "",
                relationship_level = relationship_level,
            )
            if msg is None:
                log.info("Reengagement user=%s пропущен — тихие часы МСК", user_id)
                return
            await bot.send_message(user_id, msg)
            log.info("Reengagement → user=%s (%dh)", user_id, hours_inactive)
        except TelegramForbiddenError:
            log.info("user=%s заблокировал бота — помечаем", user_id)
            await mark_user_blocked(user_id)
        except Exception as exc:
            log.error("Reengagement ошибка user=%s: %s", user_id, exc)


async def check_inactive_users(scheduler: AsyncIOScheduler) -> None:
    """
    Запускается каждый час. Находит кандидатов и планирует отправку
    с случайной задержкой 0–120 минут — сообщение не приходит ровно
    в :00, Алина пишет как живой человек.

    Частота reengagement:
      Premium → 6, 24, 48, 72 часа
      Free    → 24, 72 часа
    """
    from sqlalchemy import select as sa_select
    from database import User, UserPersona

    now = _now_utc()

    PREMIUM_HOURS = {6, 24, 48, 72}
    FREE_HOURS    = {24, 72}

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa_select(User, UserPersona)
            .join(UserPersona, UserPersona.user_id == User.id)
            .where(
                User.last_active <= now - timedelta(hours=6),
                User.last_active > now - timedelta(hours=73),
                UserPersona.is_active == True,
                User.is_blocked == False,
            )
        )
        rows = list(result.all())

    log.info("Reengagement: проверяем %d кандидатов", len(rows))

    for user, persona in rows:
        last_active = user.last_active
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        hours_inactive = int((now - last_active).total_seconds() / 3600)

        user_is_premium = await get_premium(user.id)
        target_hours = PREMIUM_HOURS if user_is_premium else FREE_HOURS

        if hours_inactive not in target_hours:
            continue

        # Случайная задержка 0–120 минут — Алина пишет не ровно в :00
        delay_minutes = random.randint(0, 120)
        run_at = now + timedelta(minutes=delay_minutes)

        scheduler.add_job(
            _send_reengagement,
            trigger="date",
            run_date=run_at,
            args=[
                user.id,
                user.first_name or "",
                user.user_name_given or "",
                hours_inactive,
                persona.relationship_level,
            ],
            misfire_grace_time=600,
            id=f"reeng_{user.id}_{hours_inactive}",
            replace_existing=True,  # не дублируем если уже запланировано
        )
        log.info(
            "Reengagement запланирован: user=%s (%dh) через %d мин (~%s UTC)",
            user.id, hours_inactive, delay_minutes,
            run_at.strftime("%H:%M"),
        )


# ── Запуск ────────────────────────────────────────────────────────────────────

async def main() -> None:
    await init_db()

    scheduler = AsyncIOScheduler(
        job_defaults={"misfire_grace_time": 600, "max_instances": 1}
    )
    scheduler.add_job(
        check_inactive_users,
        "interval",
        hours=1,
        args=[scheduler],
    )
    scheduler.start()
    log.info("Планировщик запущен")

    log.info("Бот запускается (polling)…")
    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        scheduler.shutdown(wait=False)
        await close_http_session()
        log.info("Бот остановлен корректно")


if __name__ == "__main__":
    asyncio.run(main())

