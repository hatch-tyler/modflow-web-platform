"""Slice cache service for on-demand head slice extraction with Redis caching.

This module provides backwards-compatible functions that delegate to the
unified cache service. Includes graceful degradation when Redis is unavailable.
"""

import json
import logging
from typing import Optional

from app.services.redis_manager import get_sync_cache_client

logger = logging.getLogger(__name__)

# Redis key prefix for cached slices
SLICE_KEY_PREFIX = "slice:"
# TTL for cached slices (1 hour)
SLICE_TTL = 3600

# Availability status cache
_redis_available: Optional[bool] = None


def _is_redis_available() -> bool:
    """Check if Redis is available."""
    global _redis_available
    if _redis_available is not None:
        return _redis_available

    try:
        client = get_sync_cache_client()
        client.ping()
        _redis_available = True
        return True
    except Exception as e:
        logger.debug(f"Redis not available: {e}")
        _redis_available = False
    return False


def _reset_redis_status() -> None:
    """Reset Redis availability status to force re-check."""
    global _redis_available
    _redis_available = None


def get_slice_cache_key(project_id: str, run_id: str, layer: int, kper: int, kstp: int) -> str:
    """Build cache key for a head slice."""
    return f"{SLICE_KEY_PREFIX}{project_id}:{run_id}:L{layer}_SP{kper}_TS{kstp}"


def get_cached_slice(
    project_id: str, run_id: str, layer: int, kper: int, kstp: int
) -> Optional[dict]:
    """
    Retrieve a cached head slice from Redis.

    Returns:
        Dict with slice data if cached, None otherwise (including if Redis unavailable)
    """
    if not _is_redis_available():
        return None

    try:
        r = get_sync_cache_client()
        key = get_slice_cache_key(project_id, run_id, layer, kper, kstp)
        data = r.get(key)

        if data:
            return json.loads(data)
    except Exception as e:
        logger.debug(f"Failed to get cached slice: {e}")
        _reset_redis_status()
    return None


def cache_slice(
    project_id: str,
    run_id: str,
    layer: int,
    kper: int,
    kstp: int,
    slice_data: dict,
    ttl: int = SLICE_TTL,
) -> bool:
    """
    Cache a head slice in Redis.

    Args:
        project_id: Project UUID
        run_id: Run UUID
        layer: Layer index
        kper: Stress period
        kstp: Time step
        slice_data: Dict with shape and data arrays
        ttl: Time to live in seconds (default 1 hour)

    Returns:
        True if cached successfully, False otherwise
    """
    if not _is_redis_available():
        return False

    try:
        r = get_sync_cache_client()
        key = get_slice_cache_key(project_id, run_id, layer, kper, kstp)
        r.setex(key, ttl, json.dumps(slice_data))
        return True
    except Exception as e:
        logger.debug(f"Failed to cache slice: {e}")
        _reset_redis_status()
        return False


def invalidate_run_cache(project_id: str, run_id: str) -> int:
    """
    Invalidate all cached slices for a run.

    Returns:
        Number of keys deleted (0 if Redis unavailable)
    """
    if not _is_redis_available():
        return 0

    try:
        r = get_sync_cache_client()
        pattern = f"{SLICE_KEY_PREFIX}{project_id}:{run_id}:*"
        keys = list(r.scan_iter(match=pattern))
        if keys:
            return r.delete(*keys)
    except Exception as e:
        logger.debug(f"Failed to invalidate run cache: {e}")
        _reset_redis_status()
    return 0


def get_timestep_index_key(project_id: str, run_id: str) -> str:
    """Build cache key for timestep index."""
    return f"ts_index:{project_id}:{run_id}"


def cache_timestep_index(
    project_id: str,
    run_id: str,
    index_data: dict,
    ttl: int = 7200,  # 2 hours
) -> bool:
    """
    Cache the timestep index (kstpkper list, times, grid info) for fast access.

    This avoids re-scanning the HDS file for every on-demand request.

    Returns:
        True if cached successfully, False otherwise
    """
    if not _is_redis_available():
        return False

    try:
        r = get_sync_cache_client()
        key = get_timestep_index_key(project_id, run_id)
        r.setex(key, ttl, json.dumps(index_data))
        return True
    except Exception as e:
        logger.debug(f"Failed to cache timestep index: {e}")
        _reset_redis_status()
        return False


def get_cached_timestep_index(project_id: str, run_id: str) -> Optional[dict]:
    """Retrieve cached timestep index."""
    if not _is_redis_available():
        return None

    try:
        r = get_sync_cache_client()
        key = get_timestep_index_key(project_id, run_id)
        data = r.get(key)

        if data:
            return json.loads(data)
    except Exception as e:
        logger.debug(f"Failed to get cached timestep index: {e}")
        _reset_redis_status()
    return None


def get_live_budget_key(project_id: str, run_id: str) -> str:
    """Build cache key for live budget data."""
    return f"live_budget:{project_id}:{run_id}"


def cache_live_budget(
    project_id: str,
    run_id: str,
    budget_data: dict,
    ttl: int = 7200,  # 2 hours
) -> bool:
    """
    Cache live budget data accumulated during simulation.

    Args:
        project_id: Project UUID
        run_id: Run UUID
        budget_data: Dict with record_names and periods
        ttl: Time to live in seconds (default 2 hours)

    Returns:
        True if cached successfully, False otherwise
    """
    if not _is_redis_available():
        return False

    try:
        r = get_sync_cache_client()
        key = get_live_budget_key(project_id, run_id)
        r.setex(key, ttl, json.dumps(budget_data))
        return True
    except Exception as e:
        logger.debug(f"Failed to cache live budget: {e}")
        _reset_redis_status()
        return False


def get_cached_live_budget(project_id: str, run_id: str) -> Optional[dict]:
    """Retrieve cached live budget data."""
    if not _is_redis_available():
        return None

    try:
        r = get_sync_cache_client()
        key = get_live_budget_key(project_id, run_id)
        data = r.get(key)

        if data:
            return json.loads(data)
    except Exception as e:
        logger.debug(f"Failed to get cached live budget: {e}")
        _reset_redis_status()
    return None
