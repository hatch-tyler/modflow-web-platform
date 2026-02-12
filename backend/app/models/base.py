"""SQLAlchemy base model and database session management.

Uses lazy initialization to prevent failures during module import when
the database isn't ready. Includes retry logic for robust startup.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import DateTime, create_engine, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

# ─── Lazy Engine Initialization ─────────────────────────────────────────────
# Engines are created on first access, not at import time, to allow the
# application to start even if the database isn't immediately available.

_async_engine = None
_async_session_factory = None
_sync_engine = None
_session_local = None


def get_async_engine():
    """Get or create the async database engine with retry logic."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _async_engine


def get_async_session_factory():
    """Get or create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


def get_sync_engine():
    """Get or create the sync database engine for Celery tasks."""
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        sync_database_url = settings.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        _sync_engine = create_engine(
            sync_database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=3,
        )
    return _sync_engine


def get_session_local():
    """Get or create the sync session factory for Celery tasks."""
    global _session_local
    if _session_local is None:
        _session_local = sessionmaker(
            bind=get_sync_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _session_local


class SessionLocalContextManager:
    """
    Context manager wrapper for sync database sessions.

    Allows using `with SessionLocal() as db:` pattern with lazy initialization.
    """

    def __init__(self):
        self._session = None

    def __enter__(self):
        factory = get_session_local()
        self._session = factory()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            if exc_type is not None:
                self._session.rollback()
            self._session.close()
        return False


def SessionLocal():
    """Create a sync database session with context manager support."""
    return SessionLocalContextManager()


# Backwards compatibility aliases
sync_engine = get_sync_engine
async_session_factory = get_async_session_factory
engine = get_async_engine


# ─── Base Model ─────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
    )


class UUIDMixin:
    """Mixin that adds a UUID primary key."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


# ─── Database Session Dependencies ──────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides a database session."""
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─── Database Initialization with Retry ─────────────────────────────────────


async def wait_for_db(max_retries: int = 30, retry_delay: float = 2.0) -> bool:
    """
    Wait for database to become available with exponential backoff.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Initial delay between retries (doubles each attempt, max 30s)

    Returns:
        True if database is available, False if max retries exceeded
    """
    engine = get_async_engine()
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Database not available after {max_retries} attempts: {e}")
                return False
            logger.warning(
                f"Database not ready (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await _async_sleep(delay)
            delay = min(delay * 1.5, 30.0)  # Exponential backoff, max 30s

    return False


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)


async def init_db() -> None:
    """Initialize database tables and add any missing columns."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add missing columns to existing tables (create_all does not alter tables)
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(conn) -> None:
    """Inspect the database and add any columns defined in models but missing from tables."""
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(conn)
    dialect = conn.engine.dialect
    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue
        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
        for col in table.columns:
            if col.name not in existing_cols:
                col_type = col.type.compile(dialect=dialect)
                nullable = "NULL" if col.nullable else "NOT NULL"
                conn.execute(
                    text(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {col_type} {nullable}'
                    )
                )


def wait_for_db_sync(max_retries: int = 30, retry_delay: float = 2.0) -> bool:
    """
    Synchronous version of wait_for_db for Celery worker startup.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Initial delay between retries

    Returns:
        True if database is available, False if max retries exceeded
    """
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            engine = get_sync_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established (sync)")
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Database not available after {max_retries} attempts: {e}")
                return False
            logger.warning(
                f"Database not ready (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)

    return False
