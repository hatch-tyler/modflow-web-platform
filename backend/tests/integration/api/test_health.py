"""Integration tests for health check endpoints.

Tests liveness, readiness, and detailed health checks.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock

from app.api.v1.health import ServiceStatus


class TestLivenessProbe:
    """Tests for the /health/live endpoint."""

    @pytest.mark.asyncio
    async def test_liveness_returns_alive(self, test_client: AsyncClient):
        """Test that liveness probe returns alive status."""
        response = await test_client.get("/api/v1/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    @pytest.mark.asyncio
    async def test_liveness_always_succeeds(self, test_client: AsyncClient):
        """Test that liveness probe succeeds even if services are down."""
        # Liveness should not depend on external services
        response = await test_client.get("/api/v1/health/live")

        assert response.status_code == 200


class TestReadinessProbe:
    """Tests for the /health/ready endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_with_healthy_database(self, test_client: AsyncClient):
        """Test readiness when database is healthy."""
        # The test_client fixture uses a working SQLite database
        response = await test_client.get("/api/v1/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readiness_returns_503_when_db_down(self, test_client: AsyncClient):
        """Test readiness returns 503 when database is unavailable."""
        with patch("app.api.v1.health.check_database") as mock_check:
            mock_check.return_value = ServiceStatus(
                name="database",
                status="unhealthy",
                message="Connection refused"
            )

            response = await test_client.get("/api/v1/health/ready")

            assert response.status_code == 503
            assert "Database not ready" in response.json()["detail"]


class TestHealthCheck:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, test_client: AsyncClient):
        """Test health check when all services are healthy."""
        response = await test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "version" in data
        assert data["database"] == "healthy"
        # Redis and MinIO are mocked as healthy in test_client fixture
        assert data["redis"] == "healthy"
        assert data["minio"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, test_client: AsyncClient):
        """Test health check returns degraded when some services are down."""
        with patch("app.api.v1.health.check_redis") as mock_redis:
            mock_redis.return_value = ServiceStatus(
                name="redis",
                status="unhealthy",
                message="Connection refused"
            )

            response = await test_client.get("/api/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["redis"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, test_client: AsyncClient):
        """Test health check returns unhealthy when all services are down."""
        with patch("app.api.v1.health.check_database") as mock_db:
            with patch("app.api.v1.health.check_redis") as mock_redis:
                with patch("app.api.v1.health.check_minio") as mock_minio:
                    mock_db.return_value = ServiceStatus(name="database", status="unhealthy")
                    mock_redis.return_value = ServiceStatus(name="redis", status="unhealthy")
                    mock_minio.return_value = ServiceStatus(name="minio", status="unhealthy")

                    response = await test_client.get("/api/v1/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "unhealthy"


class TestDetailedHealth:
    """Tests for the /health/detailed endpoint."""

    @pytest.mark.asyncio
    async def test_detailed_health_includes_service_info(self, test_client: AsyncClient):
        """Test detailed health includes service connection information."""
        response = await test_client.get("/api/v1/health/detailed")

        assert response.status_code == 200
        data = response.json()

        assert "version" in data
        assert "services" in data

        # Check database service info
        assert "database" in data["services"]
        db_info = data["services"]["database"]
        assert "status" in db_info
        assert "host" in db_info
        assert "port" in db_info

        # Check redis service info
        assert "redis" in data["services"]
        redis_info = data["services"]["redis"]
        assert "status" in redis_info

        # Check minio service info
        assert "minio" in data["services"]
        minio_info = data["services"]["minio"]
        assert "status" in minio_info

    @pytest.mark.asyncio
    async def test_detailed_health_includes_error_messages(self, test_client: AsyncClient):
        """Test detailed health includes error messages for unhealthy services."""
        with patch("app.api.v1.health.check_redis") as mock_redis:
            mock_redis.return_value = ServiceStatus(
                name="redis",
                status="unhealthy",
                message="Connection timed out after 5s"
            )

            response = await test_client.get("/api/v1/health/detailed")

            assert response.status_code == 200
            data = response.json()
            redis_info = data["services"]["redis"]
            assert redis_info["status"] == "unhealthy"
            assert redis_info["message"] == "Connection timed out after 5s"
