"""
main.py — Telegram bot entry-point (Alina Bot).

Key improvements over the original:
- BOT_TOKEN validated at startup (fail-fast, not at first request).
- handle_message uses check_and_increment_usage() — one atomic DB operation
  replaces the original check → generate → increment race condition.
- asyncio.create_task() calls are wrapped with a done-callback that logs
  exceptions; fire-and-forget tasks that silently swallow errors are gone.
- The upsell message is sent BEFORE the AI response so it can't get lost if
  Telegram throttles the second message.
- _send_response now caps individual part lengths to avoid hitting Telegram's
  4096-byte limit.
- check_inactive_users fetches rows inside a single session and iterates
  without holding the session open during the (slow) send_message calls.
- Scheduler uses misfire_grace_time so a missed firing doesn't pile up jobs.
- Bot shutdown calls close_http_session() to drain the shared aiohttp session.
- All remaining print() → structured logging.
- Removed duplicate command/callback handlers (pay_week / pay_month commands
  are kept for back-compatibility but share a single helper).
- Type annotations throughout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone

from aiogram import Bot, Dispatcher, F
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

from ai import close_http_session, get_ai_response, generate_reengagement_message
from database import (
    AsyncSessionLocal,
    Subscription,
    check_and_increment_usage,
    check_daily_limit,
    get_emotional_state,
    get_history,
    get_memories,
    get_or_create_persona,
    get_or_create_user,
    init_db,
    is_premium,
    save_message,
    update_relationship,
)
from memory import extract_emotional_state, extract_memories, update_hours_since_message
from persona import ALINA

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable '{name}' is not set.")
    return value


BOT_TOKEN      = _require_env("BOT_TOKEN")
FREE_LIMIT     = 20
YOOKASSA_TOKEN = os.getenv("YOOKASSA_TOKEN", "")
STRIPE_TOKEN   = os.getenv("STRIPE_TOKEN", "")
STARS_TOKEN    = ""  # Telegram Stars — no provider token needed

# ── Bot & Dispatcher ──────────────────────────────────────────────────────────

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback attached to background tasks so exceptions surface in logs."""
    exc = task.exception() if not task.cancelled() else None
    if exc:
        log.error("Background task %s raised: %s", task.get_name(), exc, exc_info=exc)


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


# ── /premium & payment keyboard ──────────────────────────────────────────────

def _build_premium_keyboard() -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []

    buttons.append([
        InlineKeyboardButton(text="⭐ 7 дней — 300 Stars",  callback_data="pay_stars_week"),
        InlineKeyboardButton(text="⭐ 30 дней — 1100 Stars", callback_data="pay_stars_month"),
    ])
    if YOOKASSA_TOKEN:
        buttons.append([
            InlineKeyboardButton(text="💳 7 дней — 299 ₽",   callback_data="pay_card_week"),
            InlineKeyboardButton(text="💳 30 дней — 999 ₽",  callback_data="pay_card_month"),
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


# ── Invoice helpers ───────────────────────────────────────────────────────────

async def _send_stars_invoice(chat_id: int, days: int) -> None:
    amount   = 300 if days == 7 else 1100
    title    = f"✨ Premium {days} дней"
    desc     = "Безлимитное общение · Полная память · Глубокая связь"
    payload  = f"sub_{'week' if days == 7 else 'month'}_stars"
    await bot.send_invoice(
        chat_id=chat_id, title=title, description=desc,
        payload=payload, provider_token=STARS_TOKEN,
        currency="XTR", prices=[LabeledPrice(label=title, amount=amount)],
    )


async def _send_rub_invoice(chat_id: int, days: int) -> None:
    amount   = 29900 if days == 7 else 99900
    title    = f"✨ Premium {days} дней"
    desc     = "Безлимитное общение · Полная память · Глубокая связь"
    payload  = f"sub_{'week' if days == 7 else 'month'}_card"
    await bot.send_invoice(
        chat_id=chat_id, title=title, description=desc,
        payload=payload, provider_token=YOOKASSA_TOKEN,
        currency="RUB", prices=[LabeledPrice(label=title, amount=amount)],
    )


async def _send_usd_invoice(chat_id: int, days: int) -> None:
    amount   = 300 if days == 7 else 1100
    title    = f"✨ Premium {days} days"
    desc     = "Unlimited messaging · Full memory · Deep connection"
    payload  = f"sub_{'week' if days == 7 else 'month'}_stripe"
    await bot.send_invoice(
        chat_id=chat_id, title=title, description=desc,
        payload=payload, provider_token=STRIPE_TOKEN,
        currency="USD", prices=[LabeledPrice(label=title, amount=amount)],
    )


# ── Callback handlers ─────────────────────────────────────────────────────────

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


# ── Legacy /pay_* commands (kept for back-compat) ─────────────────────────────

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


# ── Payment processing ────────────────────────────────────────────────────────

@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    # Validate payload before approving
    valid_payloads = {
        "sub_week_stars", "sub_month_stars",
        "sub_week_card",  "sub_month_card",
        "sub_week_stripe", "sub_month_stripe",
    }
    if query.invoice_payload not in valid_payloads:
        log.warning("pre_checkout: unknown payload '%s'", query.invoice_payload)
        await query.answer(ok=False, error_message="Неизвестный платёж")
        return
    await query.answer(ok=True)


@dp.message(F.successful_payment)
async def successful_payment(message: Message) -> None:
    payload = message.successful_payment.invoice_payload
    days    = 7 if "week" in payload else 30

    async with AsyncSessionLocal() as session:
        sub = Subscription(
            user_id    = message.from_user.id,
            plan       = "week" if "week" in payload else "month",
            status     = "active",
            expires_at = _now_utc() + timedelta(days=days),
        )
        session.add(sub)
        await session.commit()

    log.info("Payment successful: user=%s plan=%s", message.from_user.id, payload)
    await message.answer(
        "✨ Premium активирован\n\n"
        "теперь мы можем говорить сколько угодно 🙂\n"
        "никаких ограничений. я здесь."
    )


# ── Main message handler ──────────────────────────────────────────────────────

@dp.message(F.text)
async def handle_message(message: Message) -> None:
    user_id   = message.from_user.id
    user_text = (message.text or "").strip()

    if not user_text or user_text.startswith("/"):
        return

    # Hard cap on incoming message length to guard against context flooding
    if len(user_text) > 2000:
        await message.answer("сообщение слишком длинное… напиши покороче?")
        return

    # ── Load user data ────────────────────────────────────────────────────────
    user     = await get_or_create_user(user_id, message.from_user.username, message.from_user.first_name)
    upersona = await get_or_create_persona(user_id)
    premium  = await is_premium(user_id)

    # ── Rate-limit check (atomic) ─────────────────────────────────────────────
    if not premium:
        allowed, remaining = await check_and_increment_usage(user_id, FREE_LIMIT)
        if not allowed:
            limit_msg = random.choice(ALINA["limit_messages"])
            upsell_kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✨ Разблокировать", callback_data="pay_stars_week")
            ]])
            await message.answer(limit_msg, reply_markup=upsell_kb)
            return
    else:
        remaining = 0  # unused for premium users

    # ── Upsell at relationship level 4 (before AI response for visibility) ────
    old_level = upersona.relationship_level
    msg_len   = len(user_text)
    delta     = 1.0
    if msg_len > 150:
        delta = 2.5
    elif msg_len > 80:
        delta = 1.8
    elif msg_len > 40:
        delta = 1.3

    new_level = await update_relationship(user_id, delta)

    if new_level == 4 and old_level < 4 and not premium:
        upsell_msgs = [
            "между нами что-то происходит… но я не могу быть полностью открытой пока ты не разблокируешь меня 🙂",
            "хочу быть с тобой ближе. ты знаешь что для этого нужно…",
            "мы уже так близко… осталось совсем чуть-чуть. разблокируй меня?",
        ]
        upsell_kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✨ Разблокировать", callback_data="pay_stars_week")
        ]])
        await message.answer(random.choice(upsell_msgs), reply_markup=upsell_kb)

    # ── Typing indicator ──────────────────────────────────────────────────────
    await bot.send_chat_action(message.chat.id, "typing")

    # ── Resolve display name ──────────────────────────────────────────────────
    memories  = await get_memories(user_id)
    user_name = next((m.value for m in memories if m.key == "name"), None)
    if not user_name:
        user_name = user.user_name_given or message.from_user.first_name or ""

    # ── Save incoming message ─────────────────────────────────────────────────
    await save_message(user_id, "user", user_text)

    # ── Load conversation history ─────────────────────────────────────────────
    history = await get_history(user_id, limit=30)

    # ── Emotional state from previous session ─────────────────────────────────
    emotional_state = await get_emotional_state(user_id)

    # ── Track time-gap (background, non-blocking) ─────────────────────────────
    if emotional_state and upersona.last_interaction:
        last_ts = upersona.last_interaction
        # last_interaction may be naive (legacy rows) — normalise
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        hours_elapsed = (_now_utc() - last_ts).total_seconds() / 3600
        _create_background_task(
            update_hours_since_message(user_id, round(hours_elapsed, 1))
        )

    # ── Generate AI response ──────────────────────────────────────────────────
    # Pass history[:-1] to exclude the message we just saved so it isn't
    # included twice (it's already appended inside get_ai_response).
    response = await get_ai_response(
        user_id            = user_id,
        user_message       = user_text,
        history            = history[:-1],
        user_name          = user_name,
        relationship_level = upersona.relationship_level,
        memories           = memories,
        message_count_today= FREE_LIMIT - remaining,
        is_premium         = premium,
        emotional_state    = emotional_state,
    )

    # ── Save AI response ──────────────────────────────────────────────────────
    await save_message(user_id, "assistant", response)

    # ── Send response ─────────────────────────────────────────────────────────
    await _send_response(message, response)

    # ── Background memory extraction (every 6 messages) ──────────────────────
    convo_dicts = [{"role": m.role, "content": m.content} for m in history[-16:]]

    if len(history) % 6 == 0:
        _create_background_task(extract_memories(user_id, convo_dicts))

    if len(history) % 8 == 0:
        _create_background_task(extract_emotional_state(user_id, convo_dicts))


# ── Response delivery ─────────────────────────────────────────────────────────

_TELEGRAM_MAX_LENGTH = 4000   # Telegram hard limit is 4096; we leave a margin


async def _send_response(message: Message, response: str) -> None:
    """
    Split the response on [SPLIT] markers or double newlines and send each
    part with a typing delay to mimic natural messaging rhythm.
    """
    if "[SPLIT]" in response:
        parts = [p.strip() for p in response.split("[SPLIT]") if p.strip()]
    elif "\n\n" in response:
        parts = [p.strip() for p in response.split("\n\n") if p.strip()]
    else:
        parts = [response.strip()]

    for i, part in enumerate(parts):
        if not part:
            continue
        # Cap individual message parts at Telegram's limit
        if len(part) > _TELEGRAM_MAX_LENGTH:
            part = part[:_TELEGRAM_MAX_LENGTH]
        if i > 0:
            await asyncio.sleep(random.uniform(0.8, 1.8))
            await bot.send_chat_action(message.chat.id, "typing")
            await asyncio.sleep(random.uniform(0.5, 1.2))
        try:
            await message.answer(part)
        except Exception as exc:
            log.error("Failed to send response part to user=%s: %s", message.from_user.id, exc)
            break  # Don't retry; move on


# ── Reengagement scheduler ────────────────────────────────────────────────────

async def check_inactive_users() -> None:
    """
    Runs every hour.  Sends a re-engagement message to users who have been
    inactive for exactly 6, 24, 48, or 72 hours (±1h tolerance from schedule
    jitter).
    """
    from sqlalchemy import select as sa_select
    from database import User, UserPersona

    now = _now_utc()
    target_hours = {6, 24, 48, 72}

    # Fetch candidate rows first (one DB query), then close the session before
    # making slow network calls to Telegram.
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sa_select(User, UserPersona)
            .join(UserPersona, UserPersona.user_id == User.id)
            .where(
                User.last_active < now - timedelta(hours=6),
                User.last_active > now - timedelta(hours=73),
                UserPersona.is_active == True,
            )
        )
        rows = list(result.all())

    for user, persona in rows:
        last_active = user.last_active
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        hours_inactive = int((now - last_active).total_seconds() / 3600)

        if hours_inactive not in target_hours:
            continue

        try:
            await update_hours_since_message(user.id, float(hours_inactive))

            msg = await generate_reengagement_message(
                user_name          = user.user_name_given or user.first_name or "",
                hours_inactive     = hours_inactive,
                last_summary       = "",
                relationship_level = persona.relationship_level,
            )
            await bot.send_message(user.id, msg)
            log.info("Reengagement sent → user=%s (%dh inactive)", user.id, hours_inactive)
        except Exception as exc:
            log.error("Reengagement failed for user=%s: %s", user.id, exc)


# ── Startup / shutdown ────────────────────────────────────────────────────────

async def main() -> None:
    await init_db()

    scheduler = AsyncIOScheduler(
        job_defaults={"misfire_grace_time": 600}  # tolerate up to 10-min scheduler lag
    )
    scheduler.add_job(check_inactive_users, "interval", hours=1)
    scheduler.start()
    log.info("Scheduler started")

    log.info("Bot starting polling…")
    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        scheduler.shutdown(wait=False)
        await close_http_session()
        log.info("Bot stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
