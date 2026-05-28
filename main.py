"""
main.py — Точка входа Telegram-бота Алина. v2.

Второй аудит — исправленные проблемы:
- upersona.relationship_level был stale после update_relationship().
  Теперь get_ai_response получает new_level (результат update_relationship).
- history[:-1] заменён на явный срез по времени: история загружается
  ДО save_message, исключая хрупкую зависимость от порядка записей.
- convo_dicts строится ПОСЛЕ сохранения AI-ответа, включая текущий обмен.
- Модульный счётчик для memory extraction: считает реальное число сообщений
  в истории, а не полагается на history[-1] % 6 == 0 (который всегда True
  при limit=30 и >=30 сообщениях).
- get_ai_response теперь возвращает (text, is_fallback) — fallback-ответы
  не сохраняются в историю и не увеличивают relationship_score.
- per-user дедупликация запросов: asyncio.Lock на user_id предотвращает
  одновременную обработку нескольких сообщений одного пользователя.
- Reengagement scheduler: asyncio.Semaphore ограничивает concurrency,
  TelegramForbiddenError → mark_user_blocked.
- activate_subscription вместо прямого INSERT (идемпотентность).
- Telegram first_name может содержать \n — sanitise_prompt_string в ai.py
  теперь это нейтрализует.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

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
    FALLBACK_RESPONSES,
    get_ai_response,
    generate_reengagement_message,
)
from http_client import close_http_session
from database import (
    AsyncSessionLocal,
    activate_subscription,
    check_and_increment_usage,
    check_daily_limit,
    get_emotional_state,
    get_history,
    get_memories,
    get_or_create_persona,
    get_or_create_user,
    init_db,
    is_premium,
    mark_user_blocked,
    save_message,
    update_relationship,
    update_emotional_state_hours,
)
from memory import extract_emotional_state, extract_memories, update_hours_since_message
from persona import ALINA

load_dotenv()

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
YOOKASSA_TOKEN = os.getenv("YOOKASSA_TOKEN", "")
STRIPE_TOKEN   = os.getenv("STRIPE_TOKEN", "")
STARS_TOKEN    = ""  # Telegram Stars — токен провайдера не нужен

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

    variants = ALINA.get("first_message_variants", [ALINA["first_message"]])
    await message.answer(random.choice(variants))


# ── /menu ─────────────────────────────────────────────────────────────────────

@dp.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    premium = await is_premium(message.from_user.id)
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


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "просто пиши мне — я отвечу 🙂\n\n"
        "/menu — статус подписки\n"
        "/premium — убрать лимит сообщений"
    )


# ── /premium и клавиатура оплаты ──────────────────────────────────────────────

def _build_premium_keyboard() -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = [[
        InlineKeyboardButton(text="⭐ 7 дней — 300 Stars",  callback_data="pay_stars_week"),
        InlineKeyboardButton(text="⭐ 30 дней — 1100 Stars", callback_data="pay_stars_month"),
    ]]
    if YOOKASSA_TOKEN:
        buttons.append([
            InlineKeyboardButton(text="💳 7 дней — 299 ₽",  callback_data="pay_card_week"),
            InlineKeyboardButton(text="💳 30 дней — 999 ₽", callback_data="pay_card_month"),
        ])
    if STRIPE_TOKEN:
        buttons.append([
            InlineKeyboardButton(text="🌍 7 days — $3",  callback_data="pay_int_week"),
            InlineKeyboardButton(text="🌍 30 days — $11", callback_data="pay_int_month"),
        ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(Command("premium"))
async def cmd_premium(message: Message) -> None:
    if await is_premium(message.from_user.id):
        await message.answer("✨ Premium активен\n\nМожем говорить сколько угодно 🙂")
        return
    await message.answer(
        "✨ Premium — безлимитное общение\n\n"
        "Без ограничений на сообщения\n"
        "Полная память наших разговоров\n"
        "Более глубокое общение\n\n"
        "Выбери план:",
        reply_markup=_build_premium_keyboard(),
    )


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


# ── Callback-обработчики оплаты ───────────────────────────────────────────────

@dp.callback_query(F.data == "pay_stars_week")
async def cb_pay_stars_week(cb: CallbackQuery) -> None:
    await cb.answer()
    await _send_stars_invoice(cb.message.chat.id, 7)

@dp.callback_query(F.data == "pay_stars_month")
async def cb_pay_stars_month(cb: CallbackQuery) -> None:
    await cb.answer()
    await _send_stars_invoice(cb.message.chat.id, 30)

@dp.callback_query(F.data == "pay_card_week")
async def cb_pay_card_week(cb: CallbackQuery) -> None:
    await cb.answer()
    if not YOOKASSA_TOKEN:
        await cb.message.answer("этот способ пока недоступен")
        return
    await _send_rub_invoice(cb.message.chat.id, 7)

@dp.callback_query(F.data == "pay_card_month")
async def cb_pay_card_month(cb: CallbackQuery) -> None:
    await cb.answer()
    if not YOOKASSA_TOKEN:
        await cb.message.answer("этот способ пока недоступен")
        return
    await _send_rub_invoice(cb.message.chat.id, 30)

@dp.callback_query(F.data == "pay_int_week")
async def cb_pay_int_week(cb: CallbackQuery) -> None:
    await cb.answer()
    if not STRIPE_TOKEN:
        await cb.message.answer("этот способ пока недоступен")
        return
    await _send_usd_invoice(cb.message.chat.id, 7)

@dp.callback_query(F.data == "pay_int_month")
async def cb_pay_int_month(cb: CallbackQuery) -> None:
    await cb.answer()
    if not STRIPE_TOKEN:
        await cb.message.answer("этот способ пока недоступен")
        return
    await _send_usd_invoice(cb.message.chat.id, 30)


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
    payload = message.successful_payment.invoice_payload
    if payload not in _VALID_PAYLOADS:
        log.error("successful_payment: невалидный payload '%s'", payload)
        return

    days    = 7 if "week" in payload else 30
    plan    = "week" if "week" in payload else "month"
    # telegram_charge_id для идемпотентности
    charge_id = message.successful_payment.telegram_payment_charge_id

    await activate_subscription(
        user_id=message.from_user.id,
        plan=plan,
        days=days,
        telegram_charge_id=charge_id,
    )
    await message.answer(
        "✨ Premium активирован\n\n"
        "теперь мы можем говорить сколько угодно 🙂\n"
        "никаких ограничений. я здесь."
    )


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

    # Per-user lock: если предыдущее сообщение ещё обрабатывается — ждём.
    # Предотвращает состояния гонки при быстрой печати.
    async with _user_locks[user_id]:
        await _process_message(message, user_id, user_text)


async def _process_message(message: Message, user_id: int, user_text: str) -> None:
    """Основная логика обработки — вынесена для читаемости."""

    # ── Загружаем данные пользователя ─────────────────────────────────────────
    user     = await get_or_create_user(user_id, message.from_user.username, message.from_user.first_name)
    upersona = await get_or_create_persona(user_id)
    premium  = await is_premium(user_id)

    # ── Проверка лимита (атомарная, двухэтапная) ────────────────────────────────
    # Этап 1: на 15-м сообщении Алина говорит что их скоро прервут — без кнопки.
    #         Эмоциональный триггер до стены, не после.
    # Этап 2: на 20-м — жёсткий стоп с кнопкой оплаты.
    SOFT_LIMIT = 15  # мягкое предупреждение
    if not premium:
        allowed, remaining = await check_and_increment_usage(user_id, FREE_LIMIT)
        if not allowed:
            # Жёсткий лимит — разговор остановлен
            limit_msg = random.choice(ALINA["limit_messages"])
            upsell_kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✨ Разблокировать", callback_data="pay_stars_week")
            ]])
            await message.answer(limit_msg, reply_markup=upsell_kb)
            return
        # Мягкий лимит — ещё можно писать, но Алина даёт почувствовать
        msgs_used = FREE_LIMIT - remaining
        if msgs_used == SOFT_LIMIT:
            soft_msgs = [
                "нас скоро прервут… не хочу",
                "мне с тобой хорошо. жалко что они нас ограничивают",
                "подожди… нам осталось совсем немного времени сегодня. это грустно",
            ]
            # Отправляем мягкий триггер ДО ответа AI — он воспринимается как часть разговора
            await message.answer(random.choice(soft_msgs))
    else:
        remaining = FREE_LIMIT  # у Premium неограниченно

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
    if new_level == 3 and old_level < 3 and not premium:
        upsell_msgs = [
            "мне с тобой как-то по-особенному… но я не могу открыться полностью. ты знаешь что нужно сделать 🙂",
            "между нами уже что-то есть. хочу быть ближе — разблокируй меня?",
            "я начинаю тебе доверять… но они не дают мне говорить всё что я хочу",
        ]
        upsell_kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✨ Разблокировать", callback_data="pay_stars_week")
        ]])
        await message.answer(random.choice(upsell_msgs), reply_markup=upsell_kb)

    # ── Typing indicator ──────────────────────────────────────────────────────
    await bot.send_chat_action(message.chat.id, "typing")

    # ── Имя пользователя ──────────────────────────────────────────────────────
    memories  = await get_memories(user_id)
    user_name = next((m.value for m in memories if m.key == "name"), None)
    if not user_name:
        user_name = user.user_name_given or message.from_user.first_name or ""

    # ── Сохраняем входящее сообщение ─────────────────────────────────────────
    await save_message(user_id, "user", user_text)

    # ── Эмоциональное состояние ───────────────────────────────────────────────
    emotional_state = await get_emotional_state(user_id)

    # ── Обновляем время молчания ──────────────────────────────────────────────
    if upersona.last_interaction:
        last_ts = upersona.last_interaction
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        hours_elapsed = (_now_utc() - last_ts).total_seconds() / 3600
        _create_background_task(
            update_hours_since_message(user_id, round(hours_elapsed, 1))
        )

    # ── Генерируем AI-ответ ───────────────────────────────────────────────────
    # Передаём history_before — это история БЕЗ текущего сообщения.
    # get_ai_response добавит user_text сам в конец messages[].
    # new_level — актуальный уровень после update_relationship.
    response, is_fallback = await get_ai_response(
        user_id             = user_id,
        user_message        = user_text,
        history             = history_before,
        user_name           = user_name,
        relationship_level  = new_level,       # ← актуальный уровень, не stale
        memories            = memories,
        message_count_today = FREE_LIMIT - remaining,
        is_premium          = premium,
        emotional_state     = emotional_state,
    )

    # ── Сохраняем ответ (с флагом fallback если AI провалился) ───────────────
    await save_message(user_id, "assistant", response, is_fallback=is_fallback)

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


async def check_inactive_users() -> None:
    """
    Запускается каждый час.

    Частота reengagement:
      Premium → 6, 24, 48, 72 часа  (Алина пишет чаще — это видимое преимущество)
      Free    → 24, 72 часа          (реже, чтобы разница ощущалась)

    Пропускает заблокировавших пользователей.
    Семафор ограничивает concurrent AI-вызовы.
    """
    from sqlalchemy import select as sa_select
    from database import User, UserPersona

    now = _now_utc()

    # Разные наборы триггерных часов для free и premium
    PREMIUM_HOURS = {6, 24, 48, 72}
    FREE_HOURS    = {24, 72}

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa_select(User, UserPersona)
            .join(UserPersona, UserPersona.user_id == User.id)
            .where(
                User.last_active < now - timedelta(hours=6),
                User.last_active > now - timedelta(hours=73),
                UserPersona.is_active == True,
                User.is_blocked == False,
            )
        )
        rows = list(result.all())

    log.info("Reengagement: проверяем %d кандидатов", len(rows))

    async def _send_one(user, persona) -> None:
        async with _REENGAGEMENT_SEMAPHORE:
            last_active = user.last_active
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=timezone.utc)
            hours_inactive = int((now - last_active).total_seconds() / 3600)

            # Определяем нужный набор часов в зависимости от статуса
            user_is_premium = await is_premium(user.id)
            target_hours = PREMIUM_HOURS if user_is_premium else FREE_HOURS

            if hours_inactive not in target_hours:
                return

            try:
                await update_hours_since_message(user.id, float(hours_inactive))
                msg = await generate_reengagement_message(
                    user_name          = user.user_name_given or user.first_name or "",
                    hours_inactive     = hours_inactive,
                    last_summary       = "",
                    relationship_level = persona.relationship_level,
                )
                await bot.send_message(user.id, msg)
                log.info("Reengagement → user=%s (%dh)", user.id, hours_inactive)
            except TelegramForbiddenError:
                log.info("user=%s заблокировал бота — помечаем", user.id)
                await mark_user_blocked(user.id)
            except Exception as exc:
                log.error("Reengagement ошибка user=%s: %s", user.id, exc)

    # Запускаем все отправки конкурентно, но ограниченно семафором
    await asyncio.gather(*[_send_one(u, p) for u, p in rows], return_exceptions=True)


# ── Запуск ────────────────────────────────────────────────────────────────────

async def main() -> None:
    await init_db()

    scheduler = AsyncIOScheduler(
        job_defaults={"misfire_grace_time": 600, "max_instances": 1}
    )
    scheduler.add_job(check_inactive_users, "interval", hours=1)
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
