"""FastAPI application factory with robust startup sequence."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_router
from app.config import get_settings
from app.rate_limit import limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup/shutdown events.

    Includes proper service initialization with retry logic.
    """
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Wait for database to be ready
    from app.models.base import wait_for_db, init_db

    db_ready = await wait_for_db(max_retries=30, retry_delay=2.0)
    if not db_ready:
        logger.error("Database not available - some features may not work")
    else:
        # Initialize database tables (in production, use Alembic migrations instead)
        if settings.debug:
            try:
                await init_db()
                logger.info("Database tables initialized (debug mode)")
            except Exception as e:
                logger.error(f"Failed to initialize database tables: {e}")

    # Initialize storage service (non-blocking)
    try:
        from app.services.storage import get_storage_service
        storage = get_storage_service()
        if storage.is_available():
            logger.info("Storage service initialized")
        else:
            logger.warning("Storage service not immediately available - will retry on first use")
    except Exception as e:
        logger.warning(f"Could not initialize storage service: {e}")

    # Check Redis availability (non-blocking)
    try:
        from app.services.cache_service import get_cache_service
        cache = get_cache_service()
        if cache._is_redis_available():
            logger.info("Redis cache service available")
        else:
            logger.warning("Redis not available - caching disabled")
    except Exception as e:
        logger.warning(f"Could not check Redis: {e}")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")
    try:
        from app.services.redis_manager import close_all_async
        await close_all_async()
        logger.info("Async Redis connection pool closed")
    except Exception as e:
        logger.warning(f"Error closing async Redis pool: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade MODFLOW web platform for 3D visualization, "
        "simulation, parameter estimation, and uncertainty analysis.",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        docs_url=f"{settings.api_v1_prefix}/docs",
        redoc_url=f"{settings.api_v1_prefix}/redoc",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware — configurable via CORS_ORIGINS env variable (comma-separated)
    default_origins = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    extra_origins = [
        o.strip() for o in settings.cors_origins.split(",") if o.strip()
    ] if settings.cors_origins else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=default_origins + extra_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Include API routes
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/")
    async def root() -> dict:
        """Root endpoint returning API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": f"{settings.api_v1_prefix}/docs",
        }

    return app


app = create_app()
