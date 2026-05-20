import os
from datetime import date, datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, BigInteger, Integer, String, Text, Float, Boolean, Date, DateTime, select, update

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ── Модели ──────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(BigInteger, primary_key=True)  # Telegram user_id
    username        = Column(String(100))
    first_name      = Column(String(100))
    user_name_given = Column(String(100))   # имя которое пользователь сам назвал
    language        = Column(String(5), default="ru")
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_active     = Column(DateTime, default=datetime.utcnow)


class UserPersona(Base):
    __tablename__ = "user_personas"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    user_id            = Column(BigInteger, nullable=False)
    persona_id         = Column(String(50), default="alina")
    relationship_level = Column(Integer, default=1)
    relationship_score = Column(Float, default=0.0)
    is_active          = Column(Boolean, default=True)
    created_at         = Column(DateTime, default=datetime.utcnow)
    last_interaction   = Column(DateTime, default=datetime.utcnow)


class Memory(Base):
    __tablename__ = "memories"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(BigInteger, nullable=False)
    persona_id   = Column(String(50), default="alina")
    key          = Column(String(100))   # например: "job", "pet", "hobby"
    value        = Column(Text)          # например: "работает дизайнером"
    created_at   = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False)
    persona_id = Column(String(50), default="alina")
    role       = Column(String(10))      # "user" или "assistant"
    content    = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyUsage(Base):
    __tablename__ = "daily_usage"

    user_id       = Column(BigInteger, primary_key=True)
    date          = Column(Date, primary_key=True)
    messages_sent = Column(Integer, default=0)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(BigInteger, nullable=False)
    plan       = Column(String(20))      # "week" или "month"
    status     = Column(String(20), default="active")
    started_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)


# ── Инициализация ────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Хелперы ──────────────────────────────────────────────

async def get_or_create_user(user_id: int, username: str = None, first_name: str = None) -> User:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(id=user_id, username=username, first_name=first_name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        else:
            await session.execute(
                update(User).where(User.id == user_id).values(last_active=datetime.utcnow())
            )
            await session.commit()
        return user


async def get_or_create_persona(user_id: int, persona_id: str = "alina") -> UserPersona:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserPersona).where(
                UserPersona.user_id == user_id,
                UserPersona.persona_id == persona_id
            )
        )
        persona = result.scalar_one_or_none()
        if not persona:
            persona = UserPersona(user_id=user_id, persona_id=persona_id)
            session.add(persona)
            await session.commit()
            await session.refresh(persona)
        return persona


async def save_message(user_id: int, role: str, content: str, persona_id: str = "alina"):
    async with AsyncSessionLocal() as session:
        msg = Message(user_id=user_id, persona_id=persona_id, role=role, content=content)
        session.add(msg)
        await session.commit()


async def get_history(user_id: int, persona_id: str = "alina", limit: int = 20) -> list:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(Message.user_id == user_id, Message.persona_id == persona_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))


async def get_memories(user_id: int, persona_id: str = "alina") -> list:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.persona_id == persona_id
            )
        )
        return result.scalars().all()


async def save_memory(user_id: int, key: str, value: str, persona_id: str = "alina"):
    async with AsyncSessionLocal() as session:
        # Обновить если уже есть, добавить если нет
        result = await session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.persona_id == persona_id,
                Memory.key == key
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.value = value
        else:
            session.add(Memory(user_id=user_id, persona_id=persona_id, key=key, value=value))
        await session.commit()


async def check_daily_limit(user_id: int, limit: int = 20) -> tuple[bool, int]:
    """Возвращает (можно_писать, сколько_осталось)"""
    async with AsyncSessionLocal() as session:
        today = date.today()
        result = await session.execute(
            select(DailyUsage).where(
                DailyUsage.user_id == user_id,
                DailyUsage.date == today
            )
        )
        usage = result.scalar_one_or_none()
        sent = usage.messages_sent if usage else 0
        return sent < limit, max(0, limit - sent)


async def increment_usage(user_id: int):
    async with AsyncSessionLocal() as session:
        today = date.today()
        result = await session.execute(
            select(DailyUsage).where(
                DailyUsage.user_id == user_id,
                DailyUsage.date == today
            )
        )
        usage = result.scalar_one_or_none()
        if usage:
            usage.messages_sent += 1
        else:
            session.add(DailyUsage(user_id=user_id, date=today, messages_sent=1))
        await session.commit()


async def is_premium(user_id: int) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.status == "active",
                Subscription.expires_at > datetime.utcnow()
            )
        )
        return result.scalar_one_or_none() is not None


async def update_relationship(user_id: int, delta: float, persona_id: str = "alina"):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserPersona).where(
                UserPersona.user_id == user_id,
                UserPersona.persona_id == persona_id
            )
        )
        persona = result.scalar_one_or_none()
        if persona:
            persona.relationship_score += delta
            persona.last_interaction = datetime.utcnow()

            # Пересчитываем уровень
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
    return 1
