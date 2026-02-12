"""Unified caching service for MODFLOW web platform.

Provides a two-tier caching strategy:
- Redis: Hot cache for frequently accessed data (slices, timeseries, HDS index)
- MinIO: Cold cache for persistent data (parameters, geometry)

All cache keys use consistent naming conventions and support run invalidation.
Includes graceful degradation when services are unavailable.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class CacheType:
    """Cache type constants with associated TTLs (in seconds)."""

    # Redis (hot) cache types
    SLICE = "slice"                  # Head slice data, TTL: 1 hour
    TIMESERIES = "ts"                # Timeseries data, TTL: 2 hours
    HDS_INDEX = "hds_idx"            # HDS record index, TTL: 24 hours
    TIMESTEP_INDEX = "ts_index"      # Timestep index, TTL: 2 hours

    # MinIO (cold) cache types
    PARAMETERS = "parameters"         # Discovered parameters, persistent
    GEOMETRY = "geometry"             # Grid geometry, persistent
    PEST_RESULTS_SUMMARY = "pest_summary"  # PEST results summary, persistent


# TTL configuration for Redis cache types
CACHE_TTLS = {
    CacheType.SLICE: 3600,            # 1 hour
    CacheType.TIMESERIES: 7200,       # 2 hours
    CacheType.HDS_INDEX: 86400,       # 24 hours
    CacheType.TIMESTEP_INDEX: 7200,   # 2 hours
}


class CacheService:
    """Unified cache service with Redis hot cache and MinIO cold cache.

    Uses lazy initialization for external service connections and includes
    graceful degradation when services are unavailable.
    """

    def __init__(self):
        self._redis = None
        self._redis_available = None  # None = unknown, True/False = tested
        self._storage = None

    @property
    def redis(self):
        """Lazy initialization of Redis client via shared pool."""
        if self._redis is None:
            from app.services.redis_manager import get_sync_client
            self._redis = get_sync_client()
        return self._redis

    @property
    def storage(self):
        """Lazy initialization of storage service."""
        if self._storage is None:
            from app.services.storage import get_storage_service
            self._storage = get_storage_service()
        return self._storage

    def _is_redis_available(self) -> bool:
        """Check if Redis is available, with caching of result."""
        if self._redis_available is not None:
            return self._redis_available

        try:
            self.redis.ping()
            self._redis_available = True
            return True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self._redis_available = False
            return False

    def _reset_redis_status(self) -> None:
        """Reset Redis availability status to force re-check."""
        self._redis_available = None

    # ─── Key Generation ─────────────────────────────────────────────────

    def _redis_key(
        self,
        cache_type: str,
        project_id: str,
        run_id: str,
        *args: Any,
    ) -> str:
        """Generate consistent Redis cache key.

        Format: {cache_type}:{project_id}:{run_id}:{args...}
        """
        parts = [cache_type, str(project_id), str(run_id)]
        parts.extend(str(a) for a in args)
        return ":".join(parts)

    def _minio_path(
        self,
        cache_type: str,
        project_id: str,
        filename: str,
    ) -> str:
        """Generate MinIO cache path.

        Format: projects/{project_id}/cache/{cache_type}/{filename}
        """
        return f"projects/{project_id}/cache/{cache_type}/{filename}"

    # ─── Redis (Hot) Cache Operations ───────────────────────────────────

    def get_redis(
        self,
        cache_type: str,
        project_id: str,
        run_id: str,
        *args: Any,
    ) -> Optional[dict]:
        """Get data from Redis cache. Returns None if unavailable or cache miss."""
        if not self._is_redis_available():
            return None

        try:
            key = self._redis_key(cache_type, project_id, run_id, *args)
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Redis get failed for {cache_type}: {e}")
            self._reset_redis_status()
        return None

    def set_redis(
        self,
        cache_type: str,
        project_id: str,
        run_id: str,
        data: dict,
        *args: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set data in Redis cache with automatic TTL. Returns success status."""
        if not self._is_redis_available():
            return False

        try:
            key = self._redis_key(cache_type, project_id, run_id, *args)
            effective_ttl = ttl or CACHE_TTLS.get(cache_type, 3600)
            self.redis.setex(key, effective_ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.debug(f"Redis set failed for {cache_type}: {e}")
            self._reset_redis_status()
            return False

    def delete_redis(
        self,
        cache_type: str,
        project_id: str,
        run_id: str,
        *args: Any,
    ) -> bool:
        """Delete specific key from Redis cache."""
        if not self._is_redis_available():
            return False

        try:
            key = self._redis_key(cache_type, project_id, run_id, *args)
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.debug(f"Redis delete failed: {e}")
            self._reset_redis_status()
            return False

    # ─── MinIO (Cold) Cache Operations ──────────────────────────────────

    def get_minio(
        self,
        cache_type: str,
        project_id: str,
        filename: str,
    ) -> Optional[dict]:
        """Get data from MinIO cache."""
        settings = get_settings()
        path = self._minio_path(cache_type, project_id, filename)

        try:
            if not self.storage.object_exists(settings.minio_bucket_models, path):
                return None
            data = self.storage.download_file(settings.minio_bucket_models, path)
            return json.loads(data)
        except Exception as e:
            logger.debug(f"MinIO get failed for {cache_type}: {e}")
            return None

    def set_minio(
        self,
        cache_type: str,
        project_id: str,
        filename: str,
        data: dict,
    ) -> bool:
        """Set data in MinIO cache."""
        settings = get_settings()
        path = self._minio_path(cache_type, project_id, filename)

        try:
            cache_data = {
                **data,
                "_cached_at": datetime.utcnow().isoformat(),
            }
            self.storage.upload_bytes(
                settings.minio_bucket_models,
                path,
                json.dumps(cache_data).encode("utf-8"),
                content_type="application/json",
            )
            return True
        except Exception as e:
            logger.debug(f"MinIO set failed for {cache_type}: {e}")
            return False

    def delete_minio(
        self,
        cache_type: str,
        project_id: str,
        filename: str,
    ) -> bool:
        """Delete specific file from MinIO cache."""
        settings = get_settings()
        path = self._minio_path(cache_type, project_id, filename)

        try:
            if self.storage.object_exists(settings.minio_bucket_models, path):
                self.storage.delete_object(settings.minio_bucket_models, path)
            return True
        except Exception as e:
            logger.debug(f"MinIO delete failed: {e}")
            return False

    # ─── High-Level Cache Operations ────────────────────────────────────

    def get_slice(
        self,
        project_id: str,
        run_id: str,
        layer: int,
        kper: int,
        kstp: int,
    ) -> Optional[dict]:
        """Get cached head slice."""
        return self.get_redis(
            CacheType.SLICE, project_id, run_id,
            f"L{layer}", f"SP{kper}", f"TS{kstp}"
        )

    def set_slice(
        self,
        project_id: str,
        run_id: str,
        layer: int,
        kper: int,
        kstp: int,
        data: dict,
    ) -> None:
        """Cache head slice."""
        self.set_redis(
            CacheType.SLICE, project_id, run_id, data,
            f"L{layer}", f"SP{kper}", f"TS{kstp}"
        )

    def get_timeseries(
        self,
        project_id: str,
        run_id: str,
        layer: int,
        row: Optional[int] = None,
        col: Optional[int] = None,
        node: Optional[int] = None,
    ) -> Optional[dict]:
        """Get cached timeseries data."""
        if node is not None:
            return self.get_redis(
                CacheType.TIMESERIES, project_id, run_id,
                f"L{layer}", f"N{node}"
            )
        return self.get_redis(
            CacheType.TIMESERIES, project_id, run_id,
            f"L{layer}", f"R{row}", f"C{col}"
        )

    def set_timeseries(
        self,
        project_id: str,
        run_id: str,
        layer: int,
        data: dict,
        row: Optional[int] = None,
        col: Optional[int] = None,
        node: Optional[int] = None,
    ) -> None:
        """Cache timeseries data."""
        if node is not None:
            self.set_redis(
                CacheType.TIMESERIES, project_id, run_id, data,
                f"L{layer}", f"N{node}"
            )
        else:
            self.set_redis(
                CacheType.TIMESERIES, project_id, run_id, data,
                f"L{layer}", f"R{row}", f"C{col}"
            )

    def get_hds_index(
        self,
        project_id: str,
        run_id: str,
    ) -> Optional[dict]:
        """Get cached HDS record index (byte offsets for each timestep)."""
        return self.get_redis(CacheType.HDS_INDEX, project_id, run_id)

    def set_hds_index(
        self,
        project_id: str,
        run_id: str,
        data: dict,
    ) -> None:
        """Cache HDS record index."""
        self.set_redis(CacheType.HDS_INDEX, project_id, run_id, data)

    def get_timestep_index(
        self,
        project_id: str,
        run_id: str,
    ) -> Optional[dict]:
        """Get cached timestep index (kstpkper list, times, grid info)."""
        return self.get_redis(CacheType.TIMESTEP_INDEX, project_id, run_id)

    def set_timestep_index(
        self,
        project_id: str,
        run_id: str,
        data: dict,
    ) -> None:
        """Cache timestep index."""
        self.set_redis(CacheType.TIMESTEP_INDEX, project_id, run_id, data)

    # ─── Run Invalidation ───────────────────────────────────────────────

    def invalidate_run(self, project_id: str, run_id: str) -> int:
        """Invalidate all cached data for a run.

        Cleans up Redis cache entries for slices, timeseries, and indices.

        Returns:
            Number of keys deleted
        """
        if not self._is_redis_available():
            return 0

        deleted = 0

        try:
            # Invalidate all Redis cache types for this run
            for cache_type in [
                CacheType.SLICE,
                CacheType.TIMESERIES,
                CacheType.HDS_INDEX,
                CacheType.TIMESTEP_INDEX,
            ]:
                pattern = f"{cache_type}:{project_id}:{run_id}:*"
                keys = list(self.redis.scan_iter(match=pattern))
                if keys:
                    deleted += self.redis.delete(*keys)

                # Also check for keys without extra args
                base_key = f"{cache_type}:{project_id}:{run_id}"
                if self.redis.exists(base_key):
                    deleted += self.redis.delete(base_key)
        except Exception as e:
            logger.warning(f"Error invalidating run cache: {e}")
            self._reset_redis_status()

        return deleted

    def invalidate_project(self, project_id: str) -> int:
        """Invalidate all cached data for a project.

        Use with caution - clears all run caches.

        Returns:
            Number of keys deleted
        """
        if not self._is_redis_available():
            return 0

        deleted = 0

        try:
            for cache_type in [
                CacheType.SLICE,
                CacheType.TIMESERIES,
                CacheType.HDS_INDEX,
                CacheType.TIMESTEP_INDEX,
            ]:
                pattern = f"{cache_type}:{project_id}:*"
                keys = list(self.redis.scan_iter(match=pattern))
                if keys:
                    deleted += self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Error invalidating project cache: {e}")
            self._reset_redis_status()

        return deleted


# ─── Singleton Management ───────────────────────────────────────────────────

_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get or create cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


def reset_cache_service() -> None:
    """Reset the cache service singleton (for testing)."""
    global _cache_service
    _cache_service = None


def wait_for_redis(max_retries: int = 30, retry_delay: float = 2.0) -> bool:
    """
    Wait for Redis to become available.

    Args:
        max_retries: Maximum number of attempts
        retry_delay: Initial delay between retries

    Returns:
        True if Redis is available, False otherwise
    """
    from app.services.redis_manager import get_sync_client
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            r = get_sync_client()
            r.ping()
            logger.info("Redis is available")
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Redis not available after {max_retries} attempts: {e}")
                return False
            logger.debug(
                f"Redis not ready (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)

    return False
