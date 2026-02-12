"""Tests for the cache service.

Tests Redis and MinIO caching operations including graceful degradation.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.services.cache_service import (
    CacheService,
    CacheType,
    CACHE_TTLS,
    get_cache_service,
    reset_cache_service,
    wait_for_redis,
)


class TestCacheServiceKeyGeneration:
    """Tests for cache key and path generation."""

    @pytest.fixture
    def cache_service(self):
        """Create a cache service with mocked dependencies."""
        service = CacheService()
        service._redis_available = True
        service._redis = MagicMock()
        return service

    def test_redis_key_format(self, cache_service):
        """Test Redis key format generation."""
        key = cache_service._redis_key(
            CacheType.SLICE,
            "project-123",
            "run-456",
            "L0", "SP1", "TS2"
        )

        assert key == "slice:project-123:run-456:L0:SP1:TS2"

    def test_redis_key_without_args(self, cache_service):
        """Test Redis key without extra arguments."""
        key = cache_service._redis_key(
            CacheType.HDS_INDEX,
            "project-123",
            "run-456"
        )

        assert key == "hds_idx:project-123:run-456"

    def test_minio_path_format(self, cache_service):
        """Test MinIO path format generation."""
        path = cache_service._minio_path(
            CacheType.PARAMETERS,
            "project-123",
            "discovered.json"
        )

        assert path == "projects/project-123/cache/parameters/discovered.json"


class TestCacheServiceRedisOperations:
    """Tests for Redis cache operations."""

    @pytest.fixture
    def cache_service(self, fake_redis):
        """Create a cache service with FakeRedis."""
        service = CacheService()
        service._redis = fake_redis
        service._redis_available = True
        return service

    def test_get_redis_hit(self, cache_service):
        """Test successful Redis cache hit."""
        # Set up cache
        key = "slice:proj:run:L0:SP0:TS0"
        cache_service._redis.set(key, json.dumps({"heads": [[1, 2], [3, 4]]}))

        result = cache_service.get_redis(
            CacheType.SLICE, "proj", "run", "L0", "SP0", "TS0"
        )

        assert result == {"heads": [[1, 2], [3, 4]]}

    def test_get_redis_miss(self, cache_service):
        """Test Redis cache miss."""
        result = cache_service.get_redis(
            CacheType.SLICE, "proj", "run", "L0", "SP0", "TS0"
        )

        assert result is None

    def test_set_redis_with_ttl(self, cache_service):
        """Test setting Redis cache with TTL."""
        result = cache_service.set_redis(
            CacheType.SLICE, "proj", "run",
            {"heads": [[1, 2]]},
            "L0", "SP0", "TS0"
        )

        assert result is True

        # Verify data was stored
        stored = cache_service.get_redis(
            CacheType.SLICE, "proj", "run", "L0", "SP0", "TS0"
        )
        assert stored == {"heads": [[1, 2]]}

    def test_set_redis_custom_ttl(self, cache_service):
        """Test setting Redis cache with custom TTL."""
        result = cache_service.set_redis(
            CacheType.SLICE, "proj", "run",
            {"data": "test"},
            ttl=7200  # 2 hours
        )

        assert result is True

    def test_delete_redis(self, cache_service):
        """Test Redis cache deletion."""
        # Set up cache
        cache_service.set_redis(
            CacheType.SLICE, "proj", "run",
            {"heads": [[1, 2]]},
            "L0"
        )

        result = cache_service.delete_redis(
            CacheType.SLICE, "proj", "run", "L0"
        )

        assert result is True

        # Verify deletion
        stored = cache_service.get_redis(
            CacheType.SLICE, "proj", "run", "L0"
        )
        assert stored is None

    def test_graceful_degradation_when_redis_unavailable(self, cache_service):
        """Test that operations fail gracefully when Redis is unavailable."""
        cache_service._redis_available = False

        # All operations should return failure indicators
        assert cache_service.get_redis(CacheType.SLICE, "p", "r") is None
        assert cache_service.set_redis(CacheType.SLICE, "p", "r", {}) is False
        assert cache_service.delete_redis(CacheType.SLICE, "p", "r") is False


class TestCacheServiceMinioOperations:
    """Tests for MinIO cache operations."""

    @pytest.fixture
    def cache_service(self, mock_storage):
        """Create a cache service with mock storage."""
        service = CacheService()
        service._storage = mock_storage
        return service

    def test_get_minio_hit(self, cache_service, mock_storage):
        """Test successful MinIO cache hit."""
        # Set up cache
        with patch("app.services.cache_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(minio_bucket_models="models")

            path = "projects/proj/cache/parameters/params.json"
            mock_storage.upload_bytes(
                "models", path,
                json.dumps({"hk": 10.0}).encode()
            )

            result = cache_service.get_minio(
                CacheType.PARAMETERS, "proj", "params.json"
            )

            assert result == {"hk": 10.0}

    def test_get_minio_miss(self, cache_service):
        """Test MinIO cache miss."""
        with patch("app.services.cache_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(minio_bucket_models="models")

            result = cache_service.get_minio(
                CacheType.PARAMETERS, "proj", "nonexistent.json"
            )

            assert result is None

    def test_set_minio(self, cache_service, mock_storage):
        """Test setting MinIO cache."""
        with patch("app.services.cache_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(minio_bucket_models="models")

            result = cache_service.set_minio(
                CacheType.PARAMETERS, "proj", "params.json",
                {"hk": 10.0, "ss": 0.001}
            )

            assert result is True

            # Verify data was stored
            stored = cache_service.get_minio(
                CacheType.PARAMETERS, "proj", "params.json"
            )
            assert stored["hk"] == 10.0
            assert "_cached_at" in stored

    def test_delete_minio(self, cache_service, mock_storage):
        """Test MinIO cache deletion."""
        with patch("app.services.cache_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(minio_bucket_models="models")

            # Set up cache
            cache_service.set_minio(
                CacheType.PARAMETERS, "proj", "params.json",
                {"data": "test"}
            )

            result = cache_service.delete_minio(
                CacheType.PARAMETERS, "proj", "params.json"
            )

            assert result is True


class TestCacheServiceHighLevelOperations:
    """Tests for high-level cache convenience methods."""

    @pytest.fixture
    def cache_service(self, fake_redis):
        """Create a cache service with FakeRedis."""
        service = CacheService()
        service._redis = fake_redis
        service._redis_available = True
        return service

    def test_get_set_slice(self, cache_service):
        """Test slice cache operations."""
        cache_service.set_slice(
            "proj", "run", layer=0, kper=0, kstp=0,
            data={"heads": [[1, 2], [3, 4]], "min": 1, "max": 4}
        )

        result = cache_service.get_slice("proj", "run", 0, 0, 0)

        assert result == {"heads": [[1, 2], [3, 4]], "min": 1, "max": 4}

    def test_get_set_timeseries_row_col(self, cache_service):
        """Test timeseries cache with row/col coordinates."""
        cache_service.set_timeseries(
            "proj", "run", layer=0,
            data={"times": [1, 2, 3], "heads": [50, 49, 48]},
            row=10, col=20
        )

        result = cache_service.get_timeseries("proj", "run", 0, row=10, col=20)

        assert result == {"times": [1, 2, 3], "heads": [50, 49, 48]}

    def test_get_set_timeseries_node(self, cache_service):
        """Test timeseries cache with node ID (unstructured grids)."""
        cache_service.set_timeseries(
            "proj", "run", layer=0,
            data={"times": [1, 2], "heads": [50, 49]},
            node=100
        )

        result = cache_service.get_timeseries("proj", "run", 0, node=100)

        assert result == {"times": [1, 2], "heads": [50, 49]}

    def test_get_set_hds_index(self, cache_service):
        """Test HDS index cache operations."""
        index_data = {
            "records": [
                {"kstpkper": [0, 0], "offset": 0, "size": 1000},
                {"kstpkper": [0, 1], "offset": 1000, "size": 1000},
            ],
            "nlay": 3,
            "nrow": 100,
            "ncol": 100,
        }

        cache_service.set_hds_index("proj", "run", index_data)
        result = cache_service.get_hds_index("proj", "run")

        assert result == index_data

    def test_get_set_timestep_index(self, cache_service):
        """Test timestep index cache operations."""
        index_data = {
            "timesteps": [
                {"kstpkper": [0, 0], "totim": 1.0},
                {"kstpkper": [0, 1], "totim": 30.0},
            ],
            "nlay": 3,
        }

        cache_service.set_timestep_index("proj", "run", index_data)
        result = cache_service.get_timestep_index("proj", "run")

        assert result == index_data


class TestCacheServiceInvalidation:
    """Tests for cache invalidation."""

    @pytest.fixture
    def cache_service(self, fake_redis):
        """Create a cache service with FakeRedis."""
        service = CacheService()
        service._redis = fake_redis
        service._redis_available = True
        return service

    def test_invalidate_run(self, cache_service):
        """Test invalidating all cache for a run."""
        # Set up multiple cache entries
        cache_service.set_slice("proj", "run1", 0, 0, 0, {"data": 1})
        cache_service.set_slice("proj", "run1", 0, 0, 1, {"data": 2})
        cache_service.set_slice("proj", "run2", 0, 0, 0, {"data": 3})
        cache_service.set_hds_index("proj", "run1", {"index": True})

        deleted = cache_service.invalidate_run("proj", "run1")

        # run1 entries should be gone
        assert cache_service.get_slice("proj", "run1", 0, 0, 0) is None
        assert cache_service.get_slice("proj", "run1", 0, 0, 1) is None
        assert cache_service.get_hds_index("proj", "run1") is None

        # run2 entries should remain
        assert cache_service.get_slice("proj", "run2", 0, 0, 0) == {"data": 3}

    def test_invalidate_project(self, cache_service):
        """Test invalidating all cache for a project."""
        # Set up entries for multiple projects
        cache_service.set_slice("proj1", "run1", 0, 0, 0, {"data": 1})
        cache_service.set_slice("proj1", "run2", 0, 0, 0, {"data": 2})
        cache_service.set_slice("proj2", "run1", 0, 0, 0, {"data": 3})

        deleted = cache_service.invalidate_project("proj1")

        # proj1 entries should be gone
        assert cache_service.get_slice("proj1", "run1", 0, 0, 0) is None
        assert cache_service.get_slice("proj1", "run2", 0, 0, 0) is None

        # proj2 entries should remain
        assert cache_service.get_slice("proj2", "run1", 0, 0, 0) == {"data": 3}

    def test_invalidate_when_redis_unavailable(self, cache_service):
        """Test invalidation returns 0 when Redis is unavailable."""
        cache_service._redis_available = False

        assert cache_service.invalidate_run("proj", "run") == 0
        assert cache_service.invalidate_project("proj") == 0


class TestCacheServiceSingleton:
    """Tests for cache service singleton management."""

    def test_get_cache_service_creates_singleton(self):
        """Test that get_cache_service creates a singleton."""
        reset_cache_service()

        service1 = get_cache_service()
        service2 = get_cache_service()

        assert service1 is service2

        reset_cache_service()

    def test_reset_cache_service_clears_singleton(self):
        """Test that reset_cache_service clears the singleton."""
        reset_cache_service()

        service1 = get_cache_service()
        reset_cache_service()
        service2 = get_cache_service()

        assert service1 is not service2

        reset_cache_service()


class TestCacheTTLConfiguration:
    """Tests for cache TTL configuration."""

    def test_cache_ttls_defined(self):
        """Test that TTLs are defined for cache types."""
        assert CacheType.SLICE in CACHE_TTLS
        assert CacheType.TIMESERIES in CACHE_TTLS
        assert CacheType.HDS_INDEX in CACHE_TTLS
        assert CacheType.TIMESTEP_INDEX in CACHE_TTLS

    def test_slice_ttl_is_1_hour(self):
        """Test slice cache TTL is 1 hour."""
        assert CACHE_TTLS[CacheType.SLICE] == 3600

    def test_hds_index_ttl_is_24_hours(self):
        """Test HDS index cache TTL is 24 hours."""
        assert CACHE_TTLS[CacheType.HDS_INDEX] == 86400
