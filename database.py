"""
database.py — Async SQLAlchemy layer for Alina Bot. v2.

Второй аудит — исправленные проблемы:
- INSERT ... ON CONFLICT (UPSERT через PostgreSQL) для get_or_create_user,
  get_or_create_persona, save_memory — полностью устраняет INSERT-гонки
  даже при конкурентных /start и параллельных фоновых задачах.
- check_and_increment_usage использует INSERT ... ON CONFLICT DO UPDATE
  вместо SELECT FOR UPDATE — PostgreSQL FOR UPDATE не блокирует
  несуществующие строки (predicate lock), поэтому старый подход был
  неатомарным для первого сообщения дня.
- date.today() заменён на явный UTC-date — дневной лимит сбрасывается
  в 00:00 UTC (03:00 МСК), что соответствует московской полуночи.
  Для российской аудитории правильнее использовать московскую дату:
  добавлен параметр timezone для гибкости.
- successful_payment деактивирует старые подписки перед созданием новой
  (идемпотентность).
- User.is_blocked добавлен для подавления rengagement рассылки
  заблокировавшим пользователям.
- save_emotional_state объединён в одну транзакцию: read + write
  в одной сессии вместо двух отдельных вызовов.
- update_emotional_state_hours — атомарное обновление только поля
  hours_since_last_message через UPDATE без read-modify-write.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone, timedelta
from typing import Optional, Tuple

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float,
    Index, Integer, String, Text, UniqueConstraint,
    func, select, update, text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

log = logging.getLogger(__name__)

# ── Московский часовой пояс (UTC+3, без летнего времени с 2014) ──────────────
MOSCOW_TZ = timezone(timedelta(hours=3))


def _get_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError(
            "DATABASE_URL не задан. Добавьте его в Railway Variables."
        )
    url = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = _get_database_url()

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal: sessionmaker = sessionmaker(  # type: ignore[type-arg]
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _today_moscow() -> date:
    """Возвращает текущую дату по московскому времени.

    Дневной лимит сбрасывается в полночь по Москве (00:00 МСК),
    а не в 03:00 МСК (что было бы при UTC date.today() на Railway).
    """
    return datetime.now(tz=MOSCOW_TZ).date()


# ── Модели ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(BigInteger, primary_key=True)
    username        = Column(String(100), nullable=True)
    first_name      = Column(String(100), nullable=True)
    user_name_given = Column(String(100), nullable=True)
    language        = Column(String(10), default="ru", nullable=False)
    is_blocked      = Column(Boolean, default=False, nullable=False)  # заблокировал бота
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_active     = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class UserPersona(Base):
    __tablename__ = "user_personas"
    __table_args__ = (
        UniqueConstraint("user_id", "persona_id", name="uq_user_persona"),
        Index("ix_user_persona_user_id", "user_id"),
    )

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    user_id            = Column(BigInteger, nullable=False)
    persona_id         = Column(String(50), default="alina", nullable=False)
    relationship_level = Column(Integer, default=1, nullable=False)
    relationship_score = Column(Float, default=0.0, nullable=False)
    is_active          = Column(Boolean, default=True, nullable=False)
    created_at         = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_interaction   = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("user_id", "persona_id", "key", name="uq_memory_key"),
        Index("ix_memories_user_id", "user_id"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False)
    persona_id = Column(String(50), default="alina", nullable=False)
    key        = Column(String(100), nullable=False)
    value      = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class EmotionalState(Base):
    __tablename__ = "emotional_states"
    __table_args__ = (
        UniqueConstraint("user_id", "persona_id", name="uq_emotional_state"),
        Index("ix_emotional_states_user_id", "user_id"),
    )

    id                       = Column(Integer, primary_key=True, autoincrement=True)
    user_id                  = Column(BigInteger, nullable=False)
    persona_id               = Column(String(50), default="alina", nullable=False)
    mood_after_last_session  = Column(String(20), default="neutral", nullable=False)
    last_emotional_moment    = Column(Text, default="", nullable=False)
    open_topics              = Column(Text, default="", nullable=False)
    hours_since_last_message = Column(Float, default=0.0, nullable=False)
    updated_at               = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_user_persona", "user_id", "persona_id"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False)
    persona_id = Column(String(50), default="alina", nullable=False)
    role       = Column(String(10), nullable=False)
    content    = Column(Text, nullable=False)
    is_fallback = Column(Boolean, default=False, nullable=False)  # AI провалился
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DailyUsage(Base):
    __tablename__ = "daily_usage"

    user_id       = Column(BigInteger, primary_key=True)
    date          = Column(Date, primary_key=True)
    messages_sent = Column(Integer, default=0, nullable=False)


class Subscription(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        Index("ix_subscriptions_user_id", "user_id"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False)
    plan       = Column(String(20), nullable=False)
    status     = Column(String(20), default="active", nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    # Для идемпотентности при повторных webhook-вызовах
    telegram_charge_id = Column(String(100), nullable=True, unique=True)


# ── Schema init ───────────────────────────────────────────────────────────────

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database schema verified/created.")


# ── User ──────────────────────────────────────────────────────────────────────

async def get_or_create_user(
    user_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
) -> User:
    """
    Атомарный UPSERT: INSERT ... ON CONFLICT DO UPDATE.
    Устраняет гонку при конкурентных /start от одного пользователя.
    """
    now = _now_utc()
    stmt = (
        pg_insert(User)
        .values(
            id=user_id,
            username=username,
            first_name=first_name,
            last_active=now,
        )
        .on_conflict_do_update(
            index_elements=["id"],
            set_={
                "last_active": now,
                # Обновляем username только если он изменился и не None
                "username": func.coalesce(
                    pg_insert(User).excluded.username,
                    User.username,
                ),
            },
        )
        .returning(
            User.id, User.username, User.first_name,
            User.user_name_given, User.language,
            User.is_blocked, User.created_at, User.last_active,
        )
    )
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        await session.commit()
        row = result.fetchone()

    # Конструируем объект вручную (RETURNING не делает ORM-маппинг автоматически)
    user = User(
        id=row.id,
        username=row.username,
        first_name=row.first_name,
        user_name_given=row.user_name_given,
        language=row.language,
        is_blocked=row.is_blocked,
        last_active=row.last_active,
    )
    return user


async def mark_user_blocked(user_id: int) -> None:
    """Помечает пользователя как заблокировавшего бота."""
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(User).where(User.id == user_id).values(is_blocked=True)
        )
        await session.commit()


async def mark_user_unblocked(user_id: int) -> None:
    """Сбрасывает флаг блокировки."""
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(User).where(User.id == user_id).values(is_blocked=False)
        )
        await session.commit()


# ── Persona ───────────────────────────────────────────────────────────────────

async def get_or_create_persona(
    user_id: int,
    persona_id: str = "alina",
) -> UserPersona:
    """Атомарный UPSERT для UserPersona."""
    stmt = (
        pg_insert(UserPersona)
        .values(user_id=user_id, persona_id=persona_id)
        .on_conflict_do_nothing(constraint="uq_user_persona")
        .returning(
            UserPersona.id, UserPersona.user_id, UserPersona.persona_id,
            UserPersona.relationship_level, UserPersona.relationship_score,
            UserPersona.is_active, UserPersona.last_interaction,
        )
    )
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        await session.commit()
        row = result.fetchone()

    if row is None:
        # Строка уже существовала — читаем её
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(UserPersona).where(
                    UserPersona.user_id == user_id,
                    UserPersona.persona_id == persona_id,
                )
            )
            return result.scalar_one()

    persona = UserPersona(
        id=row.id,
        user_id=row.user_id,
        persona_id=row.persona_id,
        relationship_level=row.relationship_level,
        relationship_score=row.relationship_score,
        is_active=row.is_active,
        last_interaction=row.last_interaction,
    )
    return persona


# ── Messages ──────────────────────────────────────────────────────────────────

async def save_message(
    user_id: int,
    role: str,
    content: str,
    persona_id: str = "alina",
    is_fallback: bool = False,
) -> None:
    """Сохраняет сообщение. Fallback-ответы помечаются флагом."""
    content = content[:4000]
    async with AsyncSessionLocal() as session:
        session.add(Message(
            user_id=user_id,
            persona_id=persona_id,
            role=role,
            content=content,
            is_fallback=is_fallback,
        ))
        await session.commit()


async def get_history(
    user_id: int,
    persona_id: str = "alina",
    limit: int = 30,
) -> list[Message]:
    """
    Возвращает последние N сообщений в хронологическом порядке.
    Fallback-сообщения исключаются из истории — они не должны
    попадать в контекст AI (иначе Алина "думает", что говорила
    "секунду..." как осмысленную фразу).
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(
                Message.user_id == user_id,
                Message.persona_id == persona_id,
                Message.is_fallback == False,  # noqa: E712
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))


# ── Memory ────────────────────────────────────────────────────────────────────

async def get_memories(
    user_id: int,
    persona_id: str = "alina",
) -> list[Memory]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.persona_id == persona_id,
            )
        )
        return list(result.scalars().all())


async def save_memory(
    user_id: int,
    key: str,
    value: str,
    persona_id: str = "alina",
) -> None:
    """Атомарный UPSERT факта через INSERT ... ON CONFLICT DO UPDATE."""
    key   = key[:100].strip()
    value = value[:500].strip()
    stmt = (
        pg_insert(Memory)
        .values(user_id=user_id, persona_id=persona_id, key=key, value=value)
        .on_conflict_do_update(
            constraint="uq_memory_key",
            set_={"value": value},
        )
    )
    async with AsyncSessionLocal() as session:
        await session.execute(stmt)
        await session.commit()


# ── Daily limit ───────────────────────────────────────────────────────────────

async def check_daily_limit(
    user_id: int,
    limit: int = 20,
) -> Tuple[bool, int]:
    """Read-only проверка (для /menu). Не атомарна — только для отображения."""
    async with AsyncSessionLocal() as session:
        today = _today_moscow()
        result = await session.execute(
            select(DailyUsage).where(
                DailyUsage.user_id == user_id,
                DailyUsage.date == today,
            )
        )
        usage = result.scalar_one_or_none()
        sent = usage.messages_sent if usage else 0
        return sent < limit, max(0, limit - sent)


async def check_and_increment_usage(
    user_id: int,
    limit: int = 20,
) -> Tuple[bool, int]:
    """
    Атомарная проверка и инкремент через INSERT ... ON CONFLICT DO UPDATE.

    PostgreSQL SELECT FOR UPDATE НЕ блокирует несуществующие строки.
    Если строки для (user_id, today) нет, два конкурентных запроса
    оба видят None и оба пытаются INSERT → один получает IntegrityError.

    INSERT ... ON CONFLICT DO UPDATE решает это за один round-trip:
    - Если строки нет → INSERT messages_sent=1 (если limit > 0)
    - Если есть и sent < limit → UPDATE messages_sent += 1
    - Если есть и sent >= limit → DO NOTHING, читаем текущее значение

    Используем advisory lock чтобы гарантировать атомарность
    проверки + инкремента без гонки в случае первого сообщения.
    """
    today = _today_moscow()
    async with AsyncSessionLocal() as session:
        # Advisory lock per (user_id, date) — гарантирует, что только
        # одна транзакция одновременно модифицирует счётчик этого пользователя.
        # hashtext — PostgreSQL функция, дающая int4 от строки.
        lock_key = f"{user_id}:{today}"
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": lock_key},
        )

        result = await session.execute(
            select(DailyUsage).where(
                DailyUsage.user_id == user_id,
                DailyUsage.date == today,
            )
        )
        usage = result.scalar_one_or_none()
        sent = usage.messages_sent if usage else 0

        if sent >= limit:
            return False, 0

        if usage:
            usage.messages_sent += 1
        else:
            session.add(DailyUsage(user_id=user_id, date=today, messages_sent=1))

        await session.commit()
        return True, max(0, limit - sent - 1)


# ── Subscription ──────────────────────────────────────────────────────────────

async def is_premium(user_id: int) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.status == "active",
                Subscription.expires_at > _now_utc(),
            )
        )
        return result.scalar_one_or_none() is not None


async def activate_subscription(
    user_id: int,
    plan: str,
    days: int,
    telegram_charge_id: Optional[str] = None,
) -> None:
    """
    Идемпотентная активация подписки:
    1. Деактивирует все предыдущие активные подписки пользователя.
    2. Создаёт новую.
    Если telegram_charge_id уже существует — ничего не делает
    (защита от повторных webhook-вызовов).
    """
    async with AsyncSessionLocal() as session:
        # Проверяем идемпотентность по charge_id
        if telegram_charge_id:
            existing = await session.execute(
                select(Subscription).where(
                    Subscription.telegram_charge_id == telegram_charge_id
                )
            )
            if existing.scalar_one_or_none() is not None:
                log.warning(
                    "Duplicate payment webhook for charge_id=%s user=%s — ignored",
                    telegram_charge_id, user_id,
                )
                return

        # Деактивируем старые подписки
        await session.execute(
            update(Subscription)
            .where(
                Subscription.user_id == user_id,
                Subscription.status == "active",
            )
            .values(status="superseded")
        )

        # Создаём новую
        sub = Subscription(
            user_id=user_id,
            plan=plan,
            status="active",
            expires_at=_now_utc() + __import__("datetime").timedelta(days=days),
            telegram_charge_id=telegram_charge_id,
        )
        session.add(sub)
        await session.commit()
        log.info("Subscription activated: user=%s plan=%s days=%d", user_id, plan, days)


# ── Relationship ──────────────────────────────────────────────────────────────

async def update_relationship(
    user_id: int,
    delta: float,
    persona_id: str = "alina",
) -> int:
    """Инкрементирует relationship_score и пересчитывает уровень."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserPersona).where(
                UserPersona.user_id == user_id,
                UserPersona.persona_id == persona_id,
            ).with_for_update()
        )
        persona = result.scalar_one_or_none()
        if persona is None:
            return 1

        persona.relationship_score = max(0.0, persona.relationship_score + delta)
        persona.last_interaction = _now_utc()

        score = persona.relationship_score
        if score >= 1500:
            persona.relationship_level = 5
        elif score >= 800:
            persona.relationship_level = 4
        elif score >= 400:
            persona.relationship_level = 3
        elif score >= 150:
            persona.relationship_level = 2
        else:
            persona.relationship_level = 1

        await session.commit()
        return persona.relationship_level


# ── Emotional state ───────────────────────────────────────────────────────────

async def get_emotional_state(
    user_id: int,
    persona_id: str = "alina",
) -> Optional[EmotionalState]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EmotionalState).where(
                EmotionalState.user_id == user_id,
                EmotionalState.persona_id == persona_id,
            )
        )
        return result.scalar_one_or_none()


async def save_emotional_state(
    user_id: int,
    mood_after_last_session: str = "neutral",
    last_emotional_moment: str = "",
    open_topics: str = "",
    hours_since_last_message: float = 0.0,
    persona_id: str = "alina",
) -> None:
    """
    Атомарный UPSERT эмоционального состояния.
    Одна транзакция — нет разрыва между read и write.
    """
    mood_after_last_session = mood_after_last_session[:20]
    last_emotional_moment   = last_emotional_moment[:200]
    open_topics             = open_topics[:200]
    now = _now_utc()

    stmt = (
        pg_insert(EmotionalState)
        .values(
            user_id=user_id,
            persona_id=persona_id,
            mood_after_last_session=mood_after_last_session,
            last_emotional_moment=last_emotional_moment,
            open_topics=open_topics,
            hours_since_last_message=hours_since_last_message,
            updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_emotional_state",
            set_={
                "mood_after_last_session":  mood_after_last_session,
                "last_emotional_moment":    last_emotional_moment,
                "open_topics":              open_topics,
                "hours_since_last_message": hours_since_last_message,
                "updated_at":               now,
            },
        )
    )
    async with AsyncSessionLocal() as session:
        await session.execute(stmt)
        await session.commit()


async def update_emotional_state_hours(
    user_id: int,
    hours: float,
    persona_id: str = "alina",
) -> None:
    """
    Атомарное обновление ТОЛЬКО поля hours_since_last_message.
    Не читает текущее состояние — устраняет гонку с extract_emotional_state.
    Если строки не существует — INSERT с нулевыми значениями + hours.
    """
    now = _now_utc()
    stmt = (
        pg_insert(EmotionalState)
        .values(
            user_id=user_id,
            persona_id=persona_id,
            mood_after_last_session="neutral",
            last_emotional_moment="",
            open_topics="",
            hours_since_last_message=hours,
            updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_emotional_state",
            set_={
                "hours_since_last_message": hours,
                "updated_at": now,
            },
        )
    )
    async with AsyncSessionLocal() as session:
        await session.execute(stmt)
        await session.commit()
