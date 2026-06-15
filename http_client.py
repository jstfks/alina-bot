"""
http_client.py — Единая shared aiohttp-сессия для всего бота.

Выделена в отдельный модуль чтобы разорвать циклический импорт:
  ai.py      → импортировал memory.py
  memory.py  → импортировал ai.py         ← петля

Теперь:
  ai.py     → импортирует http_client.py  (нет петли)
  memory.py → импортирует http_client.py  (нет петли)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

_http_session: Optional[aiohttp.ClientSession] = None
_http_session_lock = asyncio.Lock()


async def get_http_session() -> aiohttp.ClientSession:
    """Потокобезопасная ленивая инициализация общей HTTP-сессии."""
    global _http_session
    if _http_session is not None and not _http_session.closed:
        return _http_session
    async with _http_session_lock:
        if _http_session is None or _http_session.closed:
            connector = aiohttp.TCPConnector(
                limit=50,               # макс. одновременных соединений всего
                limit_per_host=20,      # макс. на один хост (openrouter.ai)
                ttl_dns_cache=300,      # DNS кэш 5 мин
                keepalive_timeout=30,   # keep-alive 30 сек
                enable_cleanup_closed=True,
            )
            _http_session = aiohttp.ClientSession(connector=connector)
            log.debug("HTTP session создана (pool: limit=50, per_host=20)")
    return _http_session


async def close_http_session() -> None:
    """Вызывается при остановке бота."""
    global _http_session
    async with _http_session_lock:
        if _http_session and not _http_session.closed:
            await _http_session.close()
            _http_session = None
            log.info("HTTP session закрыта")
