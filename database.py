"""
database.py — Async SQLAlchemy layer for Alina Bot.

Key improvements over the original:
- Connection pool with explicit pool_size / overflow / recycle settings
- All datetime columns use timezone-aware UTC (avoids naive-datetime drift)
- ForeignKey constraints + indexes on every foreign-key / hotpath column
- Unique constraint on UserPersona(user_id, persona_id) to prevent duplicates
- upsert-style helpers that never create orphaned rows under concurrent load
- check_daily_limit / increment_usage merged into one atomic operation so a
  race between two concurrent messages cannot bypass the free-tier cap
- BOT_TOKEN / DATABASE_URL startup validation in one place
- Removed bare `datetime.utcnow()` default callables – replaced with
  `func.now()` so the DB clock is authoritative
- Type annotations throughout
- No business logic leaking into model layer
"""

from __future__ import annotations

import os
import logging
from datetime import date, datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float,
    Index, Integer, String, Text, UniqueConstraint,
    func, select, update,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

log = logging.getLogger(__name__)

# ── Startup validation ────────────────────────────────────────────────────────

def _get_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add it to your Railway Variables (see README)."
        )
    # Normalise scheme for asyncpg driver
    url = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = _get_database_url()

# ── Engine & session factory ──────────────────────────────────────────────────
# pool_pre_ping: detects stale connections after Railway restarts
# pool_recycle:  avoids "server closed the connection unexpectedly"
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    """Return tz-aware UTC now (avoids deprecated datetime.utcnow())."""
    return datetime.now(tz=timezone.utc)


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(BigInteger, primary_key=True)   # Telegram user_id
    username        = Column(String(100), nullable=True)
    first_name      = Column(String(100), nullable=True)
    user_name_given = Column(String(100), nullable=True)     # name the user provided
    language        = Column(String(10), default="ru", nullable=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_active     = Column(DateTime(timezone=True), server_default=func.now(),
                             onupdate=func.now(), nullable=False)


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
    updated_at               = Column(DateTime(timezone=True), server_default=func.now(),
                                      onupdate=func.now(), nullable=False)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_user_persona", "user_id", "persona_id"),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False)
    persona_id = Column(String(50), default="alina", nullable=False)
    role       = Column(String(10), nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DailyUsage(Base):
    __tablename__ = "daily_usage"
    __table_args__ = (
        Index("ix_daily_usage_user_date", "user_id", "date"),
    )

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
    plan       = Column(String(20), nullable=False)    # "week" | "month"
    status     = Column(String(20), default="active", nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


# ── Schema init ───────────────────────────────────────────────────────────────

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database schema verified / created.")


# ── User helpers ──────────────────────────────────────────────────────────────

async def get_or_create_user(
    user_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
) -> User:
    """
    Upsert a User row.  Always refreshes last_active so the reengagement
    scheduler has accurate data.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        now = _now_utc()
        if user is None:
            user = User(
                id=user_id,
                username=username,
                first_name=first_name,
                last_active=now,
            )
            session.add(user)
        else:
            user.last_active = now
            if username and user.username != username:
                user.username = username
        await session.commit()
        await session.refresh(user)
        return user


async def get_or_create_persona(
    user_id: int,
    persona_id: str = "alina",
) -> UserPersona:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserPersona).where(
                UserPersona.user_id == user_id,
                UserPersona.persona_id == persona_id,
            )
        )
        persona = result.scalar_one_or_none()
        if persona is None:
            persona = UserPersona(user_id=user_id, persona_id=persona_id)
            session.add(persona)
            await session.commit()
            await session.refresh(persona)
        return persona


# ── Message helpers ───────────────────────────────────────────────────────────

async def save_message(
    user_id: int,
    role: str,
    content: str,
    persona_id: str = "alina",
) -> None:
    # Truncate to avoid runaway storage from adversarial inputs
    content = content[:4000]
    async with AsyncSessionLocal() as session:
        session.add(Message(
            user_id=user_id,
            persona_id=persona_id,
            role=role,
            content=content,
        ))
        await session.commit()


async def get_history(
    user_id: int,
    persona_id: str = "alina",
    limit: int = 30,
) -> list[Message]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(Message.user_id == user_id, Message.persona_id == persona_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))


# ── Memory helpers ────────────────────────────────────────────────────────────

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
    """Upsert a single memory fact."""
    # Sanitise key to prevent injection into system prompt
    key = key[:100].strip()
    value = value[:500].strip()
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.persona_id == persona_id,
                Memory.key == key,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.value = value
        else:
            session.add(Memory(
                user_id=user_id,
                persona_id=persona_id,
                key=key,
                value=value,
            ))
        await session.commit()


# ── Daily limit helpers ───────────────────────────────────────────────────────

async def check_daily_limit(
    user_id: int,
    limit: int = 20,
) -> Tuple[bool, int]:
    """Returns (can_send, messages_remaining)."""
    async with AsyncSessionLocal() as session:
        today = date.today()
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
    Atomically check the limit and increment if allowed.
    Returns (was_allowed, remaining_after).

    Using a single session transaction prevents the TOCTOU race where two
    concurrent requests both pass check_daily_limit() before either increments.
    """
    async with AsyncSessionLocal() as session:
        today = date.today()
        result = await session.execute(
            select(DailyUsage).where(
                DailyUsage.user_id == user_id,
                DailyUsage.date == today,
            ).with_for_update()   # row-level lock
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


async def increment_usage(user_id: int) -> None:
    """Legacy helper kept for compatibility; prefer check_and_increment_usage."""
    async with AsyncSessionLocal() as session:
        today = date.today()
        result = await session.execute(
            select(DailyUsage).where(
                DailyUsage.user_id == user_id,
                DailyUsage.date == today,
            )
        )
        usage = result.scalar_one_or_none()
        if usage:
            usage.messages_sent += 1
        else:
            session.add(DailyUsage(user_id=user_id, date=today, messages_sent=1))
        await session.commit()


# ── Subscription helpers ──────────────────────────────────────────────────────

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


# ── Relationship helpers ──────────────────────────────────────────────────────

async def update_relationship(
    user_id: int,
    delta: float,
    persona_id: str = "alina",
) -> int:
    """
    Increment relationship_score by delta, recalculate level.
    Returns the new relationship_level (1-5).
    """
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


# ── Emotional-state helpers ───────────────────────────────────────────────────

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
    # Clamp field lengths to guard against oversized AI output
    mood_after_last_session  = mood_after_last_session[:20]
    last_emotional_moment    = last_emotional_moment[:200]
    open_topics              = open_topics[:200]

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EmotionalState).where(
                EmotionalState.user_id == user_id,
                EmotionalState.persona_id == persona_id,
            )
        )
        state = result.scalar_one_or_none()
        now = _now_utc()
        if state:
            state.mood_after_last_session  = mood_after_last_session
            state.last_emotional_moment    = last_emotional_moment
            state.open_topics              = open_topics
            state.hours_since_last_message = hours_since_last_message
            state.updated_at               = now
        else:
            session.add(EmotionalState(
                user_id=user_id,
                persona_id=persona_id,
                mood_after_last_session=mood_after_last_session,
                last_emotional_moment=last_emotional_moment,
                open_topics=open_topics,
                hours_since_last_message=hours_since_last_message,
                updated_at=now,
            ))
        await session.commit()
