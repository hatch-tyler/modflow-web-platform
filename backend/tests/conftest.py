"""
Shared pytest fixtures for MODFLOW Web Platform backend tests.

Provides:
- Database fixtures (async SQLite for unit tests, PostgreSQL testcontainer for integration)
- Mock services (FakeRedis, Mock MinIO storage)
- FastAPI test client with dependency overrides
- Sample data factories
"""

import asyncio
import io
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Set test environment variables BEFORE importing app modules
# Only set defaults if not already set (allows overriding via environment)
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test_db")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("MINIO_HOST", "localhost")
os.environ.setdefault("MINIO_PORT", "9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test")
os.environ.setdefault("MINIO_SECRET_KEY", "test")
os.environ.setdefault("DEBUG", "false")

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

from app.config import Settings, get_settings
from app.models.base import Base, get_db
from app.models.project import ModelType, Project, Run, RunStatus, RunType

# Check if we should use PostgreSQL (for integration tests)
USE_POSTGRES = os.environ.get("USE_POSTGRES", "false").lower() == "true"


class SQLiteUUID(TypeDecorator):
    """UUID type that works with SQLite by storing as string."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid.UUID(value)
        return value


# Override PostgreSQL-specific types for SQLite compatibility
# This only happens when not using PostgreSQL
def _override_pg_types_for_sqlite():
    """Replace PostgreSQL-specific column types with SQLite-compatible ones."""
    use_postgres = os.environ.get("USE_POSTGRES", "false").lower() == "true"
    if use_postgres:
        return  # Skip for PostgreSQL - it supports these types natively

    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, JSONB):
                column.type = JSON()
            elif isinstance(column.type, PG_UUID):
                column.type = SQLiteUUID()


_override_pg_types_for_sqlite()


# ─── Test Settings ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test settings with predictable values."""
    return Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="test",
        postgres_password="test",
        postgres_db="test_db",
        redis_host="localhost",
        redis_port=6379,
        minio_host="localhost",
        minio_port=9000,
        minio_access_key="test",
        minio_secret_key="test",
        debug=False,
    )


# ─── SQLite UUID Adapter ─────────────────────────────────────────────────────

import sqlite3

# Register adapter for UUID to work with SQLite
def _adapt_uuid(val):
    return str(val)

def _convert_uuid(val):
    return uuid.UUID(val.decode()) if val else None

sqlite3.register_adapter(uuid.UUID, _adapt_uuid)
sqlite3.register_converter("UUID", _convert_uuid)


# ─── Database Engine (Supports both SQLite and PostgreSQL) ──────────────────


@pytest_asyncio.fixture
async def async_engine():
    """Create async database engine for testing.

    Uses PostgreSQL if USE_POSTGRES=true, otherwise uses SQLite.
    """
    if USE_POSTGRES:
        # Use real PostgreSQL for integration tests
        pg_host = os.environ.get("POSTGRES_HOST", "localhost")
        pg_port = os.environ.get("POSTGRES_PORT", "5432")
        pg_user = os.environ.get("POSTGRES_USER", "modflow")
        pg_pass = os.environ.get("POSTGRES_PASSWORD", "modflow_secret_change_me")
        pg_db = os.environ.get("POSTGRES_DB", "test_db")

        database_url = f"postgresql+asyncpg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            # Disable insertmanyvalues to avoid UUID sentinel matching issues
            # with asyncpg driver. This is a known SQLAlchemy/asyncpg issue.
            insertmanyvalues_page_size=1,
        )

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine

        # Cleanup - drop all tables after tests
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        await engine.dispose()
    else:
        # Use SQLite for unit tests
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine

        # Cleanup
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing.

    For PostgreSQL integration tests, we commit changes so they're visible
    across different session instances (if any bypass the override).
    For SQLite, we use rollback for isolation.
    """
    async_session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        # For PostgreSQL, we need to clean up after each test
        # For SQLite in-memory, rollback is sufficient
        if USE_POSTGRES:
            # Clean up any uncommitted changes
            await session.rollback()
        else:
            await session.rollback()


@pytest_asyncio.fixture
async def db(db_session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
    """Alias for db_session for convenience."""
    yield db_session


# ─── Mock Storage Service ────────────────────────────────────────────────────


class MockStorageService:
    """In-memory mock for MinIO storage service."""

    def __init__(self):
        self._objects: dict[str, dict[str, bytes]] = {}
        self._available = True

    def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_data,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        if bucket not in self._objects:
            self._objects[bucket] = {}
        if hasattr(file_data, 'read'):
            self._objects[bucket][object_name] = file_data.read()
        else:
            self._objects[bucket][object_name] = file_data
        return object_name

    def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        return self.upload_file(bucket, object_name, io.BytesIO(data), len(data), content_type)

    def download_file(self, bucket: str, object_name: str) -> bytes:
        if bucket not in self._objects or object_name not in self._objects[bucket]:
            raise FileNotFoundError(f"Object {object_name} not found in bucket {bucket}")
        return self._objects[bucket][object_name]

    def download_to_file(self, bucket: str, object_name: str, file_path: Path) -> None:
        data = self.download_file(bucket, object_name)
        file_path.write_bytes(data)

    def list_objects(self, bucket: str, prefix: str = "", recursive: bool = True) -> list[str]:
        if bucket not in self._objects:
            return []
        return [k for k in self._objects[bucket] if k.startswith(prefix)]

    def delete_object(self, bucket: str, object_name: str) -> None:
        if bucket in self._objects and object_name in self._objects[bucket]:
            del self._objects[bucket][object_name]

    def delete_prefix(self, bucket: str, prefix: str) -> None:
        if bucket in self._objects:
            keys_to_delete = [k for k in self._objects[bucket] if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._objects[bucket][key]

    def object_exists(self, bucket: str, object_name: str) -> bool:
        return bucket in self._objects and object_name in self._objects[bucket]

    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires_seconds: int = 3600,
    ) -> str:
        return f"http://localhost:9000/{bucket}/{object_name}?expires={expires_seconds}"

    def download_range(
        self,
        bucket: str,
        object_name: str,
        offset: int,
        length: int,
    ) -> bytes:
        data = self.download_file(bucket, object_name)
        return data[offset:offset + length]

    def get_object_size(self, bucket: str, object_name: str) -> int:
        return len(self.download_file(bucket, object_name))

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def clear(self) -> None:
        self._objects.clear()


@pytest.fixture
def mock_storage() -> MockStorageService:
    """Provide a mock storage service."""
    return MockStorageService()


# ─── Mock Redis (FakeRedis) ──────────────────────────────────────────────────


@pytest.fixture
def fake_redis():
    """Provide a FakeRedis instance for testing."""
    try:
        import fakeredis
        return fakeredis.FakeRedis(decode_responses=True)
    except ImportError:
        # Fallback to a simple dict-based mock if fakeredis not available
        return MockRedis()


class MockRedis:
    """Simple dict-based Redis mock for testing."""

    def __init__(self):
        self._data: dict[str, str] = {}
        self._ttls: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._data[key] = value
        if ex:
            self._ttls[key] = ex
        return True

    def setex(self, key: str, time: int, value: str) -> bool:
        self._data[key] = value
        self._ttls[key] = time
        return True

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    def exists(self, key: str) -> bool:
        return key in self._data

    def ping(self) -> bool:
        return True

    def scan_iter(self, match: str = "*") -> Generator[str, None, None]:
        import fnmatch
        for key in self._data:
            if fnmatch.fnmatch(key, match):
                yield key

    def close(self) -> None:
        pass

    def clear(self) -> None:
        self._data.clear()
        self._ttls.clear()


# ─── Mock Cache Service ──────────────────────────────────────────────────────


@pytest.fixture
def mock_cache_service(fake_redis, mock_storage):
    """Provide a mock cache service."""
    from app.services.cache_service import CacheService

    service = CacheService()
    service._redis = fake_redis
    service._redis_available = True
    service._storage = mock_storage
    return service


# ─── FastAPI Test Client ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def test_client(
    async_engine,
    db_session: AsyncSession,
    mock_storage: MockStorageService,
) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with dependency overrides."""
    # Reset the app's cached database engine/session factory
    # This ensures the app uses our test database, not the production one
    import app.models.base as base_module
    original_engine = base_module._async_engine
    original_session_factory = base_module._async_session_factory

    # Replace with our test engine
    base_module._async_engine = async_engine
    base_module._async_session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    from app.main import app

    # Override database dependency to use the shared session
    async def override_get_db():
        yield db_session

    # Override storage service
    def override_get_storage():
        return mock_storage

    app.dependency_overrides[get_db] = override_get_db

    # Create mock ServiceStatus objects for health checks
    from app.api.v1.health import ServiceStatus
    mock_db_status = ServiceStatus(name="database", status="healthy")
    mock_redis_status = ServiceStatus(name="redis", status="healthy")
    mock_minio_status = ServiceStatus(name="minio", status="healthy")

    # Mock all health check functions
    async def mock_check_database():
        return mock_db_status

    async def mock_check_redis():
        return mock_redis_status

    async def mock_check_minio():
        return mock_minio_status

    # Patch storage service getter and health check functions
    with patch("app.services.storage.get_storage_service", return_value=mock_storage):
        with patch("app.api.v1.health.check_database", mock_check_database):
            with patch("app.api.v1.health.check_redis", mock_check_redis):
                with patch("app.api.v1.health.check_minio", mock_check_minio):
                    async with AsyncClient(
                        transport=ASGITransport(app=app),
                        base_url="http://test",
                    ) as client:
                        yield client

    app.dependency_overrides.clear()

    # Restore original engine/session factory
    base_module._async_engine = original_engine
    base_module._async_session_factory = original_session_factory


# ─── Sample Data Factories ───────────────────────────────────────────────────


@pytest.fixture
def sample_project_data() -> dict:
    """Sample project data for testing."""
    return {
        "name": "Test Project",
        "description": "A test MODFLOW project",
    }


@pytest.fixture
def sample_project() -> Project:
    """Create a sample Project instance (not persisted)."""
    return Project(
        id=uuid.uuid4(),
        name="Test Project",
        description="A test MODFLOW project",
        model_type=ModelType.MODFLOW_6,
        nlay=3,
        nrow=100,
        ncol=100,
        nper=12,
        is_valid=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest_asyncio.fixture
async def persisted_project(db_session: AsyncSession) -> Project:
    """Create and persist a sample project."""
    # Always use UUID object - SQLAlchemy handles conversion
    project = Project(
        id=uuid.uuid4(),
        name="Persisted Test Project",
        description="A persisted test project",
        model_type=ModelType.MODFLOW_6,
        nlay=3,
        nrow=100,
        ncol=100,
        nper=12,
        is_valid=True,
    )
    db_session.add(project)
    await db_session.commit()  # Commit for PostgreSQL visibility
    await db_session.refresh(project)
    yield project
    # Cleanup after test
    try:
        await db_session.delete(project)
        await db_session.commit()
    except Exception:
        await db_session.rollback()


@pytest.fixture
def sample_run(sample_project: Project) -> Run:
    """Create a sample Run instance (not persisted)."""
    return Run(
        id=uuid.uuid4(),
        project_id=sample_project.id,
        name="Test Run",
        run_type=RunType.FORWARD,
        status=RunStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest_asyncio.fixture
async def persisted_run(db_session: AsyncSession, persisted_project: Project) -> Run:
    """Create and persist a sample run."""
    run = Run(
        id=uuid.uuid4(),
        project_id=persisted_project.id,
        name="Persisted Test Run",
        run_type=RunType.FORWARD,
        status=RunStatus.PENDING,
    )
    db_session.add(run)
    await db_session.commit()  # Commit for PostgreSQL visibility
    await db_session.refresh(run)
    yield run
    # Cleanup after test - cascades from project deletion
    try:
        await db_session.delete(run)
        await db_session.commit()
    except Exception:
        await db_session.rollback()


# ─── Test File Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_mf6_files(tmp_path: Path) -> Path:
    """Create a minimal MF6 model structure for testing."""
    model_dir = tmp_path / "mf6_model"
    model_dir.mkdir()

    # Create mfsim.nam
    (model_dir / "mfsim.nam").write_text("""
BEGIN OPTIONS
END OPTIONS

BEGIN TIMING
  TDIS6  model.tdis
END TIMING

BEGIN MODELS
  GWF6  model.nam  model
END MODELS
""")

    # Create model.nam
    (model_dir / "model.nam").write_text("""
BEGIN OPTIONS
END OPTIONS

BEGIN PACKAGES
  DIS6  model.dis  dis
  IC6   model.ic   ic
  NPF6  model.npf  npf
  OC6   model.oc   oc
END PACKAGES
""")

    # Create model.tdis
    (model_dir / "model.tdis").write_text("""
BEGIN OPTIONS
  TIME_UNITS days
END OPTIONS

BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS

BEGIN PERIODDATA
  1.0 1 1.0
END PERIODDATA
""")

    # Create model.dis
    (model_dir / "model.dis").write_text("""
BEGIN OPTIONS
  LENGTH_UNITS meters
END OPTIONS

BEGIN DIMENSIONS
  NLAY 1
  NROW 10
  NCOL 10
END DIMENSIONS

BEGIN GRIDDATA
  DELR CONSTANT 100.0
  DELC CONSTANT 100.0
  TOP CONSTANT 100.0
  BOTM CONSTANT 0.0
END GRIDDATA
""")

    # Create model.ic (initial conditions)
    (model_dir / "model.ic").write_text("""
BEGIN GRIDDATA
  STRT CONSTANT 50.0
END GRIDDATA
""")

    # Create model.npf (node property flow)
    (model_dir / "model.npf").write_text("""
BEGIN OPTIONS
END OPTIONS

BEGIN GRIDDATA
  ICELLTYPE CONSTANT 0
  K CONSTANT 10.0
END GRIDDATA
""")

    # Create model.oc (output control)
    (model_dir / "model.oc").write_text("""
BEGIN OPTIONS
  HEAD FILEOUT model.hds
END OPTIONS

BEGIN PERIOD 1
  SAVE HEAD LAST
END PERIOD
""")

    return model_dir


@pytest.fixture
def sample_observation_csv(tmp_path: Path) -> Path:
    """Create a sample observation CSV file."""
    obs_file = tmp_path / "observations.csv"
    obs_file.write_text("""WellName,Layer,Row,Col,Time,Head
MW-01,1,10,10,1.0,45.2
MW-01,1,10,10,30.0,44.8
MW-02,1,25,30,1.0,42.1
MW-02,1,25,30,30.0,41.5
""")
    return obs_file


# ─── Helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    """Specify the async backend for pytest-asyncio."""
    return "asyncio"


def create_test_zip(files: dict[str, bytes], output_path: Path) -> Path:
    """Helper to create a test ZIP file."""
    import zipfile

    with zipfile.ZipFile(output_path, 'w') as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return output_path
