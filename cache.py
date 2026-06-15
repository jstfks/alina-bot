"""
Кэширование для Alina Bot.
Использует Redis для кэширования данных, требующих частого доступа к БД.
"""

import json
import logging
import os
from typing import Any, Optional

import redis.asyncio as redis

log = logging.getLogger(__name__)

# ── Redis client ──────────────────────────────────────────────────────────────

_redis_client = None


async def _get_redis_client() -> redis.Redis:
    """Возвращает Redis-клиент, создает если не существует."""
    global _redis_client
    if _redis_client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
        try:
            await _redis_client.ping()
            log.info("Redis подключен")
        except Exception as exc:
            log.warning("Redis недоступен: %s — кэширование отключено", exc)
            _redis_client = None
    return _redis_client


async def _get_redis() -> Optional[redis.Redis]:
    """Возвращает Redis-клиент или None если Redis недоступен."""
    return await _get_redis_client()


# ── Кэширование премиум-статуса ───────────────────────────────────────────────

async def get_premium(user_id: int) -> bool:
    """
    Возвращает премиум-статус пользователя с кэшированием.
    TTL: 60 секунд.
    """
    redis_client = await _get_redis()
    if not redis_client:
        # Fallback к БД если Redis недоступен
        from database import is_premium
        return await is_premium(user_id)

    cache_key = f"premium:{user_id}"
    cached = await redis_client.get(cache_key)
    if cached is not None:
        return cached == "true"

    # Запрашиваем из БД (через is_premium для кэширования)
    from database import is_premium
    premium = await is_premium(user_id)

    # Сохраняем в кэш
    await redis_client.setex(
        cache_key,
        60,  # TTL в секундах
        "true" if premium else "false"
    )

    return premium


async def invalidate_premium_cache(user_id: int) -> None:
    """Инвалидирует кэш премиум-статуса пользователя."""
    redis_client = await _get_redis()
    if redis_client:
        cache_key = f"premium:{user_id}"
        await redis_client.delete(cache_key)


async def invalidate_all_premium_cache() -> None:
    """Инвалидирует весь кэш премиум-статуса."""
    redis_client = await _get_redis()
    if redis_client:
        pattern = "premium:*"
        keys = await redis_client.keys(pattern)
        if keys:
            await redis_client.delete(*keys)
            log.info("Инвалидирован кэш премиум-статуса: %d ключей", len(keys))


# ── Инвалидация кэша при изменении премиум-статуса ──────────────────────────────

async def on_subscription_changed(user_id: int) -> None:
    """Вызывать после активации/деактивации подписки."""
    await invalidate_premium_cache(user_id)
    log.info("[Cache] Инвалидирован кэш премиум-статуса для user=%s", user_id)