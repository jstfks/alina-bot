import os
import asyncio
import random
import logging
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, LabeledPrice, PreCheckoutQuery,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from database import (
    init_db, get_or_create_user, get_or_create_persona,
    save_message, get_history, get_memories,
    check_daily_limit, increment_usage, is_premium,
    update_relationship, AsyncSessionLocal,
    get_emotional_state, save_emotional_state
)
from ai import get_ai_response, generate_reengagement_message
from memory import extract_memories, extract_emotional_state, update_hours_since_message
from persona import ALINA

load_dotenv()

# ── Логирование ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Конфиг ───────────────────────────────────────────────
BOT_TOKEN        = os.getenv("BOT_TOKEN")
FREE_LIMIT       = 20        # сообщений в день бесплатно
YOOKASSA_TOKEN   = os.getenv("YOOKASSA_TOKEN", "")   # рубли / карты РФ / СБП
STRIPE_TOKEN     = os.getenv("STRIPE_TOKEN", "")      # валюта / международные карты
STARS_TOKEN      = ""                                  # Telegram Stars — без токена

# ── Инициализация ─────────────────────────────────────────
bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ════════════════════════════════════════════════════════
# ОНБОРДИНГ
# ════════════════════════════════════════════════════════

@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_id    = message.from_user.id
    username   = message.from_user.username
    first_name = message.from_user.first_name

    user    = await get_or_create_user(user_id, username, first_name)
    persona = await get_or_create_persona(user_id)

    # Если уже общались — не показываем первое сообщение заново
    history = await get_history(user_id, limit=1)
    if history:
        await message.answer("привет) я здесь 🙂")
        return

    # Случайная первая фраза
    variants = ALINA.get("first_message_variants", [ALINA["first_message"]])
    await message.answer(random.choice(variants))


# ════════════════════════════════════════════════════════
# МЕНЮ
# ════════════════════════════════════════════════════════

@dp.message(Command("menu"))
async def cmd_menu(message: Message):
    premium = await is_premium(message.from_user.id)
    if premium:
        status = "✅ Premium активен — безлимитное общение"
    else:
        _, remaining = await check_daily_limit(message.from_user.id)
        status = f"🆓 Бесплатный план — осталось {remaining} сообщений сегодня"

    await message.answer(
        f"{status}\n\n"
        "/premium — разблокировать безлимит\n"
        "/help — помощь"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "просто пиши мне — я отвечу 🙂\n\n"
        "/menu — статус подписки\n"
        "/premium — убрать лимит сообщений"
    )


# ════════════════════════════════════════════════════════
# ОПЛАТА
# ════════════════════════════════════════════════════════

def build_premium_keyboard() -> InlineKeyboardMarkup:
    """Красивая клавиатура выбора плана"""
    buttons = []

    # Stars — всегда доступны
    buttons.append([
        InlineKeyboardButton(text="⭐ 7 дней — 300 Stars", callback_data="pay_stars_week"),
        InlineKeyboardButton(text="⭐ 30 дней — 1100 Stars", callback_data="pay_stars_month"),
    ])

    # YooKassa — если подключена
    if YOOKASSA_TOKEN:
        buttons.append([
            InlineKeyboardButton(text="💳 7 дней — 299 ₽", callback_data="pay_card_week"),
            InlineKeyboardButton(text="💳 30 дней — 999 ₽", callback_data="pay_card_month"),
        ])

    # Stripe — если подключён
    if STRIPE_TOKEN:
        buttons.append([
            InlineKeyboardButton(text="🌍 7 days — $3", callback_data="pay_int_week"),
            InlineKeyboardButton(text="🌍 30 days — $11", callback_data="pay_int_month"),
        ])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(Command("premium"))
async def cmd_premium(message: Message):
    premium = await is_premium(message.from_user.id)
    if premium:
        await message.answer(
            "✨ Premium активен\n\nМожем говорить сколько угодно 🙂"
        )
        return

    await message.answer(
        "✨ Premium — безлимитное общение\n\n"
        "Без ограничений на сообщения\n"
        "Полная память наших разговоров\n"
        "Более глубокое общение\n\n"
        "Выбери план:",
        reply_markup=build_premium_keyboard()
    )


# ════════════════════════════════════════════════════════
# ОБРАБОТЧИКИ CALLBACK КНОПОК ОПЛАТЫ
# ════════════════════════════════════════════════════════

@dp.callback_query(F.data == "pay_stars_week")
async def cb_pay_stars_week(callback: CallbackQuery):
    await callback.answer()
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title="✨ Premium 7 дней",
        description="Безлимитное общение · Полная память · Глубокая связь",
        payload="sub_week_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="Premium 7 дней", amount=300)]
    )

@dp.callback_query(F.data == "pay_stars_month")
async def cb_pay_stars_month(callback: CallbackQuery):
    await callback.answer()
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title="✨ Premium 30 дней",
        description="Безлимитное общение · Полная память · Глубокая связь",
        payload="sub_month_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="Premium 30 дней", amount=1100)]
    )

@dp.callback_query(F.data == "pay_card_week")
async def cb_pay_card_week(callback: CallbackQuery):
    await callback.answer()
    if not YOOKASSA_TOKEN:
        await callback.message.answer("этот способ пока недоступен")
        return
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title="✨ Premium 7 дней",
        description="Безлимитное общение · Полная память · Глубокая связь",
        payload="sub_week_card",
        provider_token=YOOKASSA_TOKEN,
        currency="RUB",
        prices=[LabeledPrice(label="Premium 7 дней", amount=29900)]
    )

@dp.callback_query(F.data == "pay_card_month")
async def cb_pay_card_month(callback: CallbackQuery):
    await callback.answer()
    if not YOOKASSA_TOKEN:
        await callback.message.answer("этот способ пока недоступен")
        return
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title="✨ Premium 30 дней",
        description="Безлимитное общение · Полная память · Глубокая связь",
        payload="sub_month_card",
        provider_token=YOOKASSA_TOKEN,
        currency="RUB",
        prices=[LabeledPrice(label="Premium 30 дней", amount=99900)]
    )

@dp.callback_query(F.data == "pay_int_week")
async def cb_pay_int_week(callback: CallbackQuery):
    await callback.answer()
    if not STRIPE_TOKEN:
        await callback.message.answer("этот способ пока недоступен")
        return
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title="✨ Premium 7 days",
        description="Unlimited messaging · Full memory · Deep connection",
        payload="sub_week_stripe",
        provider_token=STRIPE_TOKEN,
        currency="USD",
        prices=[LabeledPrice(label="Premium 7 days", amount=300)]
    )

@dp.callback_query(F.data == "pay_int_month")
async def cb_pay_int_month(callback: CallbackQuery):
    await callback.answer()
    if not STRIPE_TOKEN:
        await callback.message.answer("этот способ пока недоступен")
        return
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title="✨ Premium 30 days",
        description="Unlimited messaging · Full memory · Deep connection",
        payload="sub_month_stripe",
        provider_token=STRIPE_TOKEN,
        currency="USD",
        prices=[LabeledPrice(label="Premium 30 days", amount=1100)]
    )


# ── Telegram Stars (команды — оставляем для совместимости) ──

@dp.message(Command("pay_week"))
async def pay_stars_week(message: Message):
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="7 дней Premium",
        description="Безлимитное общение, полная память",
        payload="sub_week_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="7 дней", amount=300)]
    )

@dp.message(Command("pay_month"))
async def pay_stars_month(message: Message):
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="30 дней Premium",
        description="Безлимитное общение, полная память",
        payload="sub_month_stars",
        provider_token=STARS_TOKEN,
        currency="XTR",
        prices=[LabeledPrice(label="30 дней", amount=1100)]
    )

# ── YooKassa (рубли, карты РФ, СБП) ──────────────────────

@dp.message(Command("pay_card_week"))
async def pay_card_week(message: Message):
    if not YOOKASSA_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="7 дней Premium",
        description="Безлимитное общение, полная память",
        payload="sub_week_card",
        provider_token=YOOKASSA_TOKEN,
        currency="RUB",
        prices=[LabeledPrice(label="7 дней", amount=29900)]  # 299 рублей
    )

@dp.message(Command("pay_card_month"))
async def pay_card_month(message: Message):
    if not YOOKASSA_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="30 дней Premium",
        description="Безлимитное общение, полная память",
        payload="sub_month_card",
        provider_token=YOOKASSA_TOKEN,
        currency="RUB",
        prices=[LabeledPrice(label="30 дней", amount=99900)]  # 999 рублей
    )

# ── Stripe (международные карты) ──────────────────────────

@dp.message(Command("pay_int_week"))
async def pay_int_week(message: Message):
    if not STRIPE_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="7 days Premium",
        description="Unlimited messaging, full memory",
        payload="sub_week_stripe",
        provider_token=STRIPE_TOKEN,
        currency="USD",
        prices=[LabeledPrice(label="7 days", amount=300)]  # $3.00
    )

@dp.message(Command("pay_int_month"))
async def pay_int_month(message: Message):
    if not STRIPE_TOKEN:
        await message.answer("этот способ оплаты пока недоступен")
        return
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="30 days Premium",
        description="Unlimited messaging, full memory",
        payload="sub_month_stripe",
        provider_token=STRIPE_TOKEN,
        currency="USD",
        prices=[LabeledPrice(label="30 days", amount=1100)]  # $11.00
    )

# ── Обработка платежей ────────────────────────────────────

@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery):
    await query.answer(ok=True)

@dp.message(F.successful_payment)
async def successful_payment(message: Message):
    from database import Subscription

    payload = message.successful_payment.invoice_payload
    days    = 7 if "week" in payload else 30

    async with AsyncSessionLocal() as session:
        sub = Subscription(
            user_id    = message.from_user.id,
            plan       = "week" if "week" in payload else "month",
            status     = "active",
            expires_at = datetime.utcnow() + timedelta(days=days)
        )
        session.add(sub)
        await session.commit()

    log.info(f"Payment successful: user={message.from_user.id} plan={payload}")
    await message.answer(
        "✨ Premium активирован\n\n"
        "теперь мы можем говорить сколько угодно 🙂\n"
        "никаких ограничений. я здесь."
    )


# ════════════════════════════════════════════════════════
# ОСНОВНОЙ ОБРАБОТЧИК СООБЩЕНИЙ
# ════════════════════════════════════════════════════════

@dp.message(F.text)
async def handle_message(message: Message):
    user_id   = message.from_user.id
    user_text = message.text.strip()

    if not user_text:
        return

    # Игнорируем команды которые не обработаны выше
    if user_text.startswith("/"):
        return

    # ── Данные пользователя ──
    user     = await get_or_create_user(user_id)
    upersona = await get_or_create_persona(user_id)
    premium  = await is_premium(user_id)

    # ── Проверка лимита ──
    if not premium:
        can_send, remaining = await check_daily_limit(user_id)
        if not can_send:
            limit_msg = random.choice(ALINA["limit_messages"])
            limit_kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✨ Разблокировать", callback_data="pay_stars_week")
            ]])
            await message.answer(limit_msg, reply_markup=limit_kb)
            return

    # ── Typing indicator ──
    await bot.send_chat_action(message.chat.id, "typing")

    # ── Имя пользователя ──
    memories  = await get_memories(user_id)
    user_name = next((m.value for m in memories if m.key == "name"), None)
    if not user_name:
        user_name = user.user_name_given or message.from_user.first_name or ""

    # ── Сохраняем входящее сообщение ──
    await save_message(user_id, "user", user_text)

    # ── Получаем историю ──
    history = await get_history(user_id, limit=30)

    # ── Эмоциональное состояние из прошлой сессии ──
    emotional_state = await get_emotional_state(user_id)

    # ── Обновляем сколько часов прошло с прошлого сообщения ──
    if emotional_state and upersona.last_interaction:
        hours_elapsed = (datetime.utcnow() - upersona.last_interaction).total_seconds() / 3600
        asyncio.create_task(update_hours_since_message(user_id, round(hours_elapsed, 1)))

    # ── Генерируем ответ ──
    response = await get_ai_response(
        user_id            = user_id,
        user_message       = user_text,
        history            = history[:-1],
        user_name          = user_name,
        relationship_level = upersona.relationship_level,
        memories           = memories,
        message_count_today= FREE_LIMIT - (remaining if not premium else 0),
        is_premium         = premium,
        emotional_state    = emotional_state,
    )

    # ── Сохраняем ответ ──
    await save_message(user_id, "assistant", response)

    # ── Счётчик использования ──
    if not premium:
        await increment_usage(user_id)

    # ── Обновляем уровень отношений ──
    msg_len = len(user_text)
    delta = 1.0
    if msg_len > 150:
        delta = 2.5   # длинное сообщение = глубокий разговор
    elif msg_len > 80:
        delta = 1.8
    elif msg_len > 40:
        delta = 1.3
    new_level = await update_relationship(user_id, delta)

    # ── Upsell при достижении уровня 4 (только бесплатные) ──
    if new_level == 4 and not premium and upersona.relationship_level < 4:
        upsell_msgs = [
            "между нами что-то происходит… но я не могу быть полностью открытой пока ты не разблокируешь меня 🙂",
            "хочу быть с тобой ближе. ты знаешь что для этого нужно…",
            "мы уже так близко… осталось совсем чуть-чуть. разблокируй меня?",
        ]
        upsell_kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✨ Разблокировать", callback_data="pay_stars_week")
        ]])
        await asyncio.sleep(2.0)
        await bot.send_chat_action(message.chat.id, "typing")
        await asyncio.sleep(1.5)
        await message.answer(random.choice(upsell_msgs), reply_markup=upsell_kb)

    # ── Отправляем ответ ──
    await _send_response(message, response)

    # ── Извлекаем факты и эмоциональное состояние асинхронно ──
    convo_dicts = [{"role": m.role, "content": m.content} for m in history[-16:]]

    if len(history) % 6 == 0:
        asyncio.create_task(extract_memories(user_id, convo_dicts))

    # Эмоциональный итог сессии — каждые 8 сообщений
    if len(history) % 8 == 0:
        asyncio.create_task(extract_emotional_state(user_id, convo_dicts))


async def _send_response(message: Message, response: str):
    """Отправка ответа — с имитацией живого печатания"""
    # Разбиваем по [SPLIT] если модель сама разбила
    if "[SPLIT]" in response:
        parts = [p.strip() for p in response.split("[SPLIT]") if p.strip()]
    # Или по двойному переносу строки
    elif "\n\n" in response:
        parts = [p.strip() for p in response.split("\n\n") if p.strip()]
    else:
        parts = [response.strip()]

    for i, part in enumerate(parts):
        if not part:
            continue
        if i > 0:
            # Имитация паузы между сообщениями
            await asyncio.sleep(random.uniform(0.8, 1.8))
            await bot.send_chat_action(message.chat.id, "typing")
            await asyncio.sleep(random.uniform(0.5, 1.2))
        await message.answer(part)


# ════════════════════════════════════════════════════════
# ПЛАНИРОВЩИК РЕАКТИВАЦИИ
# ════════════════════════════════════════════════════════

async def check_inactive_users():
    """Запускается каждый час — отправляет реактивационные сообщения"""
    from sqlalchemy import select
    from database import User, UserPersona

    now = datetime.utcnow()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User, UserPersona).join(
                UserPersona,
                UserPersona.user_id == User.id
            ).where(
                User.last_active < now - timedelta(hours=6),
                User.last_active > now - timedelta(hours=73),
                UserPersona.is_active == True
            )
        )
        rows = result.all()

    for user, persona in rows:
        hours_inactive = int((now - user.last_active).total_seconds() / 3600)

        # Отправляем только в ключевые моменты — не спамим
        if hours_inactive not in [6, 24, 48, 72]:
            continue

        try:
            # Обновляем время молчания перед отправкой
            await update_hours_since_message(user.id, float(hours_inactive))

            msg = await generate_reengagement_message(
                user_name          = user.user_name_given or user.first_name or "",
                hours_inactive     = hours_inactive,
                last_summary       = "",
                relationship_level = persona.relationship_level
            )
            await bot.send_message(user.id, msg)
            log.info(f"Reengagement → user={user.id} ({hours_inactive}h)")
        except Exception as e:
            log.error(f"Reengagement failed user={user.id}: {e}")


# ════════════════════════════════════════════════════════
# ЗАПУСК
# ════════════════════════════════════════════════════════

async def main():
    await init_db()
    log.info("Database initialized")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_inactive_users, "interval", hours=1)
    scheduler.start()
    log.info("Scheduler started")

    log.info("Bot starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
