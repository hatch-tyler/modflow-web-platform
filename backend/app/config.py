"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "MODFLOW Web Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Security
    secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Database
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "modflow"
    postgres_password: str = "modflow_secret"
    postgres_db: str = "modflow_db"

    @property
    def database_url(self) -> str:
        """Construct async database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Construct sync database URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = "redis_secret"

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Celery
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    def get_celery_broker_url(self) -> str:
        """Get Celery broker URL, defaulting to Redis."""
        return self.celery_broker_url or self.redis_url

    def get_celery_result_backend(self) -> str:
        """Get Celery result backend, defaulting to Redis."""
        return self.celery_result_backend or self.redis_url

    # MinIO / S3
    minio_host: str = "minio"
    minio_port: int = 9000
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_models: str = "modflow-models"
    minio_bucket_results: str = "modflow-results"

    @property
    def minio_endpoint(self) -> str:
        """Construct MinIO endpoint."""
        return f"{self.minio_host}:{self.minio_port}"

    # MODFLOW executables
    mf6_exe_path: str = "/usr/local/bin/mf6"
    mf2005_exe_path: str = "/usr/local/bin/mf2005"
    mfnwt_exe_path: str = "/usr/local/bin/mfnwt"
    mfusg_exe_path: str = "/usr/local/bin/mfusg"

    # PEST++ executables
    pestpp_exe_path: str = "/usr/local/bin/pestpp-glm"
    pestpp_ies_exe_path: str = "/usr/local/bin/pestpp-ies"

    # PEST++ parallel workers
    pest_default_num_workers: int = 4
    pest_max_num_workers: int = 32

    # PEST++ network mode (for distributed agents across machines)
    pest_network_mode: bool = False
    pest_manager_port: int = 4004
    pest_manager_host: str = "0.0.0.0"  # Bind address for manager
    pest_agent_timeout: int = 300  # Seconds to wait for agents to connect

    # PEST++ local container agents (Phase 2)
    # These run as Docker containers on the main server
    pest_local_containers: bool = False
    pest_local_agents: int = 4  # Number of local container agents
    pest_workspace_path: str = "/tmp/pest-workspace"  # Shared volume path

    # Resource limits per local agent container
    # Optimized for Intel i7-14700 (20 cores, 32GB RAM)
    # - 16 agents × 2GB = 32GB RAM
    # - 16 agents × 1 CPU = 16 cores (leaving 4 for system/manager)
    pest_agent_memory_limit: str = "2G"
    pest_agent_cpu_limit: float = 1.0
    pest_agent_memory_reservation: str = "512M"
    pest_agent_cpu_reservation: float = 0.5

    # Docker compose file for local agents
    pest_compose_file: str = "docker-compose.pest.yml"

    # File limits
    max_upload_size_mb: int = 500
    max_model_files: int = 5000  # Increased to support models with many external array files


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
