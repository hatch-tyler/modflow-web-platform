"""Health check endpoints with graceful degradation."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])
settings = get_settings()

# Timeout for individual service checks (seconds).
# All checks run concurrently, so worst-case response time ≈ this value.
_CHECK_TIMEOUT = 3.0


class ServiceStatus(BaseModel):
    """Status of an individual service."""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: Optional[str] = None


class HealthCheck(BaseModel):
    """Overall health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    database: str
    redis: str
    minio: str


async def check_database() -> ServiceStatus:
    """Check database connectivity with graceful handling."""
    try:
        async def _check():
            from app.models.base import get_async_session_factory
            factory = get_async_session_factory()
            async with factory() as session:
                await session.execute(text("SELECT 1"))
            return ServiceStatus(name="database", status="healthy")

        return await asyncio.wait_for(_check(), timeout=_CHECK_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Database health check timed out")
        return ServiceStatus(name="database", status="unhealthy", message="Timeout")
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return ServiceStatus(name="database", status="unhealthy", message=str(e))


async def check_redis() -> ServiceStatus:
    """Check Redis connectivity with graceful handling."""
    try:
        async def _check():
            from app.services.redis_manager import get_async_client
            redis_client = await get_async_client()
            await redis_client.ping()
            return ServiceStatus(name="redis", status="healthy")

        return await asyncio.wait_for(_check(), timeout=_CHECK_TIMEOUT)
    except asyncio.TimeoutError:
        logger.debug("Redis health check timed out")
        return ServiceStatus(name="redis", status="unhealthy", message="Timeout")
    except Exception as e:
        logger.debug(f"Redis health check failed: {e}")
        return ServiceStatus(name="redis", status="unhealthy", message=str(e))


async def check_minio() -> ServiceStatus:
    """Check MinIO connectivity with graceful handling."""
    try:
        async def _check():
            from app.services.storage import get_storage_service
            storage = get_storage_service()
            loop = asyncio.get_event_loop()
            available = await loop.run_in_executor(None, storage.is_available)
            if available:
                return ServiceStatus(name="minio", status="healthy")
            return ServiceStatus(name="minio", status="unhealthy", message="Connection failed")

        return await asyncio.wait_for(_check(), timeout=_CHECK_TIMEOUT)
    except asyncio.TimeoutError:
        logger.debug("MinIO health check timed out")
        return ServiceStatus(name="minio", status="unhealthy", message="Timeout")
    except Exception as e:
        logger.debug(f"MinIO health check failed: {e}")
        return ServiceStatus(name="minio", status="unhealthy", message=str(e))


@router.get("", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """
    Check health of all services.

    Returns overall health status and individual service statuses.
    The API can still function in degraded mode if some services are down.
    """
    db_status, redis_status, minio_status = await asyncio.gather(
        check_database(),
        check_redis(),
        check_minio(),
    )

    statuses = [db_status, redis_status, minio_status]
    healthy_count = sum(1 for s in statuses if s.status == "healthy")

    if healthy_count == 3:
        overall = "healthy"
    elif healthy_count >= 1:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthCheck(
        status=overall,
        version=settings.app_version,
        database=db_status.status,
        redis=redis_status.status,
        minio=minio_status.status,
    )


@router.get("/live")
async def liveness() -> dict:
    """
    Kubernetes/Docker liveness probe.

    Just confirms the application process is running and can respond to HTTP.
    This should NOT check external dependencies - that's what readiness is for.
    """
    return {"status": "alive"}


@router.get("/ready")
async def readiness() -> dict:
    """
    Kubernetes/Docker readiness probe.

    Confirms the app can handle requests. Checks critical services only.
    Returns 503 if the database is not available (required for all operations).
    """
    db_status = await check_database()

    if db_status.status != "healthy":
        raise HTTPException(
            status_code=503,
            detail=f"Database not ready: {db_status.message}"
        )

    # Redis and MinIO are optional for readiness - app can function without them
    return {"status": "ready"}


@router.get("/detailed")
@limiter.limit("60/minute")
async def detailed_health(request: Request) -> dict:
    """
    Detailed health check with full service information.

    Useful for debugging and monitoring dashboards.
    Note: host/port details are intentionally omitted to avoid
    exposing internal infrastructure information.
    """
    db_status, redis_status, minio_status = await asyncio.gather(
        check_database(),
        check_redis(),
        check_minio(),
    )

    return {
        "version": settings.app_version,
        "services": {
            "database": {
                "status": db_status.status,
                "message": db_status.message,
            },
            "redis": {
                "status": redis_status.status,
                "message": redis_status.message,
            },
            "minio": {
                "status": minio_status.status,
                "message": minio_status.message,
            },
        },
    }
