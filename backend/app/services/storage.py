"""MinIO object storage service with retry logic and graceful degradation."""

import functools
import io
import logging
import time
from pathlib import Path
from typing import BinaryIO, Optional, TypeVar, Callable
from urllib3.exceptions import HTTPError, MaxRetryError, TimeoutError as Urllib3TimeoutError

from minio import Minio
from minio.error import S3Error

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _retry_on_transient(
    max_attempts: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 15.0,
    description: str = "operation",
) -> Callable:
    """Decorator that retries MinIO operations on transient network errors.

    Retries on connection errors, timeouts, and server-side errors (5xx).
    Does NOT retry on client errors like 404 (NoSuchKey) or 403.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except S3Error as e:
                    # Only retry on server errors (5xx), not client errors (4xx)
                    if e.code and not str(e.code).startswith("5"):
                        raise
                    last_exception = e
                except (ConnectionError, OSError, HTTPError, MaxRetryError, Urllib3TimeoutError) as e:
                    last_exception = e
                except Exception as e:
                    # Don't retry unknown errors
                    raise

                if attempt < max_attempts:
                    logger.warning(
                        f"MinIO {description} failed (attempt {attempt}/{max_attempts}): "
                        f"{last_exception}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

            logger.error(f"MinIO {description} failed after {max_attempts} attempts")
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


class StorageService:
    """Service for interacting with MinIO object storage.

    Includes retry logic for transient failures and graceful initialization.
    """

    def __init__(self, max_retries: int = 5, retry_delay: float = 2.0):
        """
        Initialize storage service with retry logic.

        Args:
            max_retries: Maximum connection attempts during initialization
            retry_delay: Initial delay between retries (exponential backoff)
        """
        settings = get_settings()
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._buckets_ensured = False
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._settings = settings

        # Try to ensure buckets, but don't fail if MinIO isn't ready
        self._try_ensure_buckets()

    def _try_ensure_buckets(self) -> bool:
        """Attempt to ensure buckets exist, with retry logic."""
        if self._buckets_ensured:
            return True

        delay = self._retry_delay
        for attempt in range(1, self._max_retries + 1):
            try:
                self._ensure_buckets()
                self._buckets_ensured = True
                logger.info("MinIO buckets verified/created successfully")
                return True
            except Exception as e:
                if attempt == self._max_retries:
                    logger.warning(
                        f"Could not verify MinIO buckets after {self._max_retries} attempts: {e}. "
                        "Operations may fail until MinIO is available."
                    )
                    return False
                logger.debug(
                    f"MinIO not ready (attempt {attempt}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * 1.5, 10.0)
        return False

    def _ensure_buckets(self) -> None:
        """Ensure required buckets exist."""
        for bucket in [self._settings.minio_bucket_models, self._settings.minio_bucket_results]:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def _ensure_ready(self) -> None:
        """Ensure buckets are ready before operations."""
        if not self._buckets_ensured:
            if not self._try_ensure_buckets():
                raise RuntimeError("MinIO storage is not available")

    @_retry_on_transient(description="upload_file")
    def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload a file to MinIO.

        Args:
            bucket: Bucket name
            object_name: Object path in bucket
            file_data: File-like object to upload
            length: Size of file in bytes
            content_type: MIME type

        Returns:
            Object name/path
        """
        self._ensure_ready()
        # Seek to start in case this is a retry with the same file handle
        if hasattr(file_data, 'seek'):
            file_data.seek(0)
        self.client.put_object(
            bucket,
            object_name,
            file_data,
            length,
            content_type=content_type,
        )
        return object_name

    def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes directly to MinIO."""
        return self.upload_file(
            bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type,
        )

    @_retry_on_transient(description="download_file")
    def download_file(self, bucket: str, object_name: str) -> bytes:
        """
        Download a file from MinIO.

        Args:
            bucket: Bucket name
            object_name: Object path in bucket

        Returns:
            File contents as bytes
        """
        self._ensure_ready()
        response = self.client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    @_retry_on_transient(description="download_to_file")
    def download_to_file(
        self,
        bucket: str,
        object_name: str,
        file_path: Path,
    ) -> None:
        """Download an object to a local file."""
        self._ensure_ready()
        self.client.fget_object(bucket, object_name, str(file_path))

    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[str]:
        """List objects in a bucket with optional prefix."""
        self._ensure_ready()
        objects = self.client.list_objects(
            bucket,
            prefix=prefix,
            recursive=recursive,
        )
        return [obj.object_name for obj in objects]

    def delete_object(self, bucket: str, object_name: str) -> None:
        """Delete an object from MinIO."""
        self._ensure_ready()
        self.client.remove_object(bucket, object_name)

    def delete_prefix(self, bucket: str, prefix: str) -> None:
        """Delete all objects with a given prefix."""
        objects = self.list_objects(bucket, prefix=prefix, recursive=True)
        for obj in objects:
            self.delete_object(bucket, obj)

    def object_exists(self, bucket: str, object_name: str) -> bool:
        """Check if an object exists."""
        try:
            self._ensure_ready()
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False
        except RuntimeError:
            # MinIO not available - treat as not exists
            return False

    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Get a presigned URL for downloading an object."""
        from datetime import timedelta

        self._ensure_ready()
        return self.client.presigned_get_object(
            bucket,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )

    @_retry_on_transient(description="download_range")
    def download_range(
        self,
        bucket: str,
        object_name: str,
        offset: int,
        length: int,
    ) -> bytes:
        """
        Download a byte range from an object using HTTP Range requests.

        This is critical for large binary files (like HDS) where we only
        need specific records without downloading the entire file.

        Args:
            bucket: Bucket name
            object_name: Object path in bucket
            offset: Starting byte offset (0-indexed)
            length: Number of bytes to read

        Returns:
            Requested bytes
        """
        self._ensure_ready()
        response = self.client.get_object(
            bucket,
            object_name,
            offset=offset,
            length=length,
        )
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_object_size(self, bucket: str, object_name: str) -> int:
        """Get the size of an object in bytes."""
        self._ensure_ready()
        stat = self.client.stat_object(bucket, object_name)
        return stat.size

    def is_available(self) -> bool:
        """Check if MinIO storage is available."""
        try:
            self.client.list_buckets()
            return True
        except Exception:
            return False


# ─── Singleton Management ───────────────────────────────────────────────────

_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


def reset_storage_service() -> None:
    """Reset the storage service singleton (for testing)."""
    global _storage_service
    _storage_service = None


def wait_for_storage(max_retries: int = 30, retry_delay: float = 2.0) -> bool:
    """
    Wait for MinIO storage to become available.

    Args:
        max_retries: Maximum number of attempts
        retry_delay: Initial delay between retries

    Returns:
        True if storage is available, False otherwise
    """
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            service = get_storage_service()
            if service.is_available():
                logger.info("MinIO storage is available")
                return True
        except Exception as e:
            pass

        if attempt == max_retries:
            logger.error(f"MinIO not available after {max_retries} attempts")
            return False

        logger.debug(
            f"MinIO not ready (attempt {attempt}/{max_retries}). "
            f"Retrying in {delay:.1f}s..."
        )
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)

    return False
