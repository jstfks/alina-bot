import os
import asyncio
import random
import logging
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, LabeledPrice, PreCheckoutQuery
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from database import (
    init_db, get_or_create_user, get_or_create_persona,
    save_message, get_history, get_memories,
    check_daily_limit, increment_usage, is_premium,
    update_relationship, AsyncSessionLocal
)
from ai import get_ai_response, generate_reengagement_message
from memory import extract_memories
from persona import ALINA

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BOT_TOKEN   = os.getenv("BOT_TOKEN")
FREE_LIMIT  = 20  # сообщений в день бесплатно

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ── Онбординг ────────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_id = message.from_user.id
    username = message.from_user.username
    first_name = message.from_user.first_name

    await get_or_create_user(user_id, username, first_name)
    await get_or_create_persona(user_id)

    # Первое сообщение от Алины
    await message.answer(ALINA["first_message"])


# ── Команда /menu ─────────────────────────────────────────

@dp.message(Command("menu"))
async def cmd_menu(message: Message):
    premium = await is_premium(message.from_user.id)
    status = "✅ Premium активен" if premium else "🆓 Бесплатный план (20 сообщений/день)"
    await message.answer(
        f"{status}\n\n"
        "Команды:\n"
        "/premium — разблокировать безлимит\n"
        "/menu — это меню"
    )


# ── Оплата ───────────────────────────────────────────────

@dp.message(Command("premium"))
async def cmd_premium(message: Message):
    await message.answer(
        "выбери план:\n\n"
        "🗓 7 дней — /pay_week\n"
        "📅 30 дней — /pay_month"
    )

@dp.message(Command("pay_week"))
async def pay_week(message: Message):
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="7 дней Premium",
        description="Безлимитные сообщения, полная память, приоритет",
        payload="sub_week",
        currency="XTR",
        prices=[LabeledPrice(label="7 дней", amount=300)]
    )

@dp.message(Command("pay_month"))
async def pay_month(message: Message):
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="30 дней Premium",
        description="Безлимитные сообщения, полная память, приоритет",
        payload="sub_month",
        currency="XTR",
        prices=[LabeledPrice(label="30 дней", amount=1100)]
    )

@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery):
    await query.answer(ok=True)

@dp.message(F.successful_payment)
async def successful_payment(message: Message):
    from database import Subscription, AsyncSessionLocal
    from sqlalchemy import select

    payload = message.successful_payment.invoice_payload
    days = 7 if "week" in payload else 30

    async with AsyncSessionLocal() as session:
        sub = Subscription(
            user_id=message.from_user.id,
            plan="week" if "week" in payload else "month",
            status="active",
            expires_at=datetime.utcnow() + timedelta(days=days)
        )
        session.add(sub)
        await session.commit()

    await message.answer("окей, теперь мы можем говорить сколько угодно 🙂")


# ── Основной обработчик сообщений ────────────────────────

@dp.message(F.text)
async def handle_message(message: Message):
    user_id   = message.from_user.id
    user_text = message.text.strip()

    if not user_text:
        return

    # Получаем данные пользователя
    user   = await get_or_create_user(user_id)
    upersona = await get_or_create_persona(user_id)
    premium  = await is_premium(user_id)

    # Проверяем лимит (только для бесплатных)
    if not premium:
        can_send, remaining = await check_daily_limit(user_id)
        if not can_send:
            limit_msg = random.choice(ALINA["limit_messages"])
            await message.answer(limit_msg)
            await message.answer("👉 /premium — убрать ограничение")
            return

    # Показываем "печатает..."
    await bot.send_chat_action(message.chat.id, "typing")

    # Имя пользователя (из памяти или Telegram)
    memories = await get_memories(user_id)
    user_name = None
    for m in memories:
        if m.key == "name":
            user_name = m.value
            break
    if not user_name:
        user_name = user.user_name_given or message.from_user.first_name or ""

    # Сохраняем сообщение пользователя
    await save_message(user_id, "user", user_text)

    # Получаем историю
    history = await get_history(user_id, limit=20)

    # Генерируем ответ
    response = await get_ai_response(
        user_id=user_id,
        user_message=user_text,
        history=history[:-1],  # без текущего сообщения
        user_name=user_name,
        relationship_level=upersona.relationship_level,
        memories=memories,
        message_count_today=FREE_LIMIT - (remaining if not premium else 0)
    )

    # Сохраняем ответ
    await save_message(user_id, "assistant", response)

    # Инкрементируем счётчик
    if not premium:
        await increment_usage(user_id)

    # Обновляем уровень отношений
    msg_len = len(user_text)
    delta = 1.0
    if msg_len > 100:
        delta = 2.0
    elif msg_len > 50:
        delta = 1.5
    await update_relationship(user_id, delta)

    # Отправляем ответ
    # Если в ответе есть [SPLIT] — отправляем двумя сообщениями
    if "[SPLIT]" in response:
        parts = response.split("[SPLIT]")
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                if i > 0:
                    await asyncio.sleep(1.2)
                    await bot.send_chat_action(message.chat.id, "typing")
                    await asyncio.sleep(0.8)
                await message.answer(part)
    else:
        await message.answer(response)

    # Извлекаем память асинхронно (не блокируем ответ)
    if len(history) % 5 == 0:  # каждые 5 сообщений
        asyncio.create_task(
            extract_memories(user_id, [{"role": m.role, "content": m.content} for m in history])
        )


# ── Планировщик реактивации ──────────────────────────────

async def check_inactive_users():
    """Запускается каждый час, отправляет реактивационные сообщения"""
    from sqlalchemy import select, and_
    from database import User, UserPersona, Message

    now = datetime.utcnow()

    async with AsyncSessionLocal() as session:
        # Ищем пользователей которые не писали 6-72 часа
        result = await session.execute(
            select(User).where(
                User.last_active < now - timedelta(hours=6),
                User.last_active > now - timedelta(hours=72)
            )
        )
        inactive_users = result.scalars().all()

    for user in inactive_users:
        hours_inactive = int((now - user.last_active).total_seconds() / 3600)

        # Не спамим — один раз за период
        # (в продакшне нужна таблица sent_reengagements)
        if hours_inactive not in [6, 24, 48, 72]:
            continue

        try:
            msg = await generate_reengagement_message(
                user_name=user.user_name_given or user.first_name or "",
                hours_inactive=hours_inactive,
                last_summary="",
                relationship_level=1
            )
            await bot.send_message(user.id, msg)
            log.info(f"Reengagement sent to {user.id} ({hours_inactive}h inactive)")
        except Exception as e:
            log.error(f"Reengagement failed for {user.id}: {e}")


# ── Запуск ───────────────────────────────────────────────

async def main():
    await init_db()
    log.info("Database initialized")

    # Планировщик
    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_inactive_users, "interval", hours=1)
    scheduler.start()

    log.info("Bot starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
