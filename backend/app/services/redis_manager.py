"""Centralized Redis connection manager.

Provides a single shared ConnectionPool for all sync callers (services + Celery
tasks) and a single shared async ConnectionPool for all async callers (SSE
endpoints, health checks).  This eliminates the per-call ``redis.from_url()``
pattern that was causing TCP port exhaustion on Windows (Event 4231).

Usage:
    # Sync (Celery tasks, services)
    from app.services.redis_manager import get_sync_client
    r = get_sync_client()
    r.set("key", "value")

    # Async (FastAPI endpoints)
    from app.services.redis_manager import get_async_client
    r = await get_async_client()
    await r.set("key", "value")
"""

import logging
import threading
from typing import Optional

import redis
import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Sync pool ────────────────────────────────────────────────────────────────

_sync_pool: Optional[redis.ConnectionPool] = None
_sync_lock = threading.Lock()


def _get_sync_pool() -> redis.ConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        with _sync_lock:
            if _sync_pool is None:  # double-check after acquiring lock
                settings = get_settings()
                _sync_pool = redis.ConnectionPool.from_url(
                    settings.redis_url,
                    max_connections=20,
                    decode_responses=True,
                    socket_keepalive=True,
                    socket_connect_timeout=5,
                    health_check_interval=30,
                )
    return _sync_pool


def get_sync_client() -> redis.Redis:
    """Return a Redis client backed by the shared sync connection pool."""
    return redis.Redis(connection_pool=_get_sync_pool())


def close_all_sync() -> None:
    """Disconnect every connection in the sync pool (call on worker shutdown)."""
    global _sync_pool
    with _sync_lock:
        if _sync_pool is not None:
            try:
                _sync_pool.disconnect()
            except Exception as e:
                logger.warning(f"Error closing sync Redis pool: {e}")
            _sync_pool = None

# ── Async pool ───────────────────────────────────────────────────────────────

_async_pool: Optional[aioredis.ConnectionPool] = None
_async_lock = threading.Lock()


def _get_async_pool() -> aioredis.ConnectionPool:
    global _async_pool
    if _async_pool is None:
        with _async_lock:
            if _async_pool is None:  # double-check after acquiring lock
                settings = get_settings()
                _async_pool = aioredis.ConnectionPool.from_url(
                    settings.redis_url,
                    max_connections=50,
                    decode_responses=True,
                    socket_keepalive=True,
                    socket_connect_timeout=5,
                    health_check_interval=30,
                )
    return _async_pool


async def get_async_client() -> aioredis.Redis:
    """Return an async Redis client backed by the shared async connection pool."""
    return aioredis.Redis(connection_pool=_get_async_pool())


async def close_all_async() -> None:
    """Disconnect every connection in the async pool (call on app shutdown)."""
    global _async_pool
    with _async_lock:
        if _async_pool is not None:
            try:
                await _async_pool.disconnect()
            except Exception as e:
                logger.warning(f"Error closing async Redis pool: {e}")
            _async_pool = None
