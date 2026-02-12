"""Tests for the storage service.

Tests MinIO storage operations including upload, download, range requests, and error handling.
"""

import io
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.storage import (
    StorageService,
    get_storage_service,
    reset_storage_service,
    wait_for_storage,
)


class TestStorageServiceInitialization:
    """Tests for StorageService initialization and bucket management."""

    def test_init_creates_minio_client(self):
        """Test that initialization creates a MinIO client."""
        with patch("app.services.storage.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                minio_endpoint="localhost:9000",
                minio_access_key="test",
                minio_secret_key="test",
                minio_secure=False,
                minio_bucket_models="models",
                minio_bucket_results="results",
            )
            with patch("app.services.storage.Minio") as mock_minio:
                mock_client = MagicMock()
                mock_client.bucket_exists.return_value = True
                mock_minio.return_value = mock_client

                service = StorageService(max_retries=1, retry_delay=0.01)

                mock_minio.assert_called_once_with(
                    "localhost:9000",
                    access_key="test",
                    secret_key="test",
                    secure=False,
                )

    def test_ensure_buckets_creates_missing_buckets(self):
        """Test that missing buckets are created during initialization."""
        with patch("app.services.storage.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                minio_endpoint="localhost:9000",
                minio_access_key="test",
                minio_secret_key="test",
                minio_secure=False,
                minio_bucket_models="models",
                minio_bucket_results="results",
            )
            with patch("app.services.storage.Minio") as mock_minio:
                mock_client = MagicMock()
                mock_client.bucket_exists.return_value = False
                mock_minio.return_value = mock_client

                service = StorageService(max_retries=1, retry_delay=0.01)

                assert mock_client.make_bucket.call_count == 2

    def test_retry_logic_on_init_failure(self):
        """Test retry logic when MinIO is not available during init."""
        with patch("app.services.storage.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                minio_endpoint="localhost:9000",
                minio_access_key="test",
                minio_secret_key="test",
                minio_secure=False,
                minio_bucket_models="models",
                minio_bucket_results="results",
            )
            with patch("app.services.storage.Minio") as mock_minio:
                mock_client = MagicMock()
                # First two attempts fail, third succeeds
                # Then second bucket check also succeeds
                mock_client.bucket_exists.side_effect = [
                    Exception("Connection refused"),  # Attempt 1
                    Exception("Connection refused"),  # Attempt 2
                    True,  # Attempt 3: first bucket exists
                    True,  # Second bucket check
                ]
                mock_minio.return_value = mock_client

                service = StorageService(max_retries=3, retry_delay=0.01)

                # Should have retried (2 failures + 2 success checks for 2 buckets)
                assert mock_client.bucket_exists.call_count == 4


class TestStorageServiceOperations:
    """Tests for StorageService file operations."""

    @pytest.fixture
    def storage_service(self):
        """Create a storage service with mocked MinIO client."""
        with patch("app.services.storage.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                minio_endpoint="localhost:9000",
                minio_access_key="test",
                minio_secret_key="test",
                minio_secure=False,
                minio_bucket_models="models",
                minio_bucket_results="results",
            )
            with patch("app.services.storage.Minio") as mock_minio:
                mock_client = MagicMock()
                mock_client.bucket_exists.return_value = True
                mock_minio.return_value = mock_client

                service = StorageService(max_retries=1, retry_delay=0.01)
                service._mock_client = mock_client  # Expose for assertions
                yield service

    def test_upload_file(self, storage_service):
        """Test file upload."""
        file_data = io.BytesIO(b"test content")

        result = storage_service.upload_file(
            bucket="models",
            object_name="test/file.txt",
            file_data=file_data,
            length=12,
            content_type="text/plain",
        )

        assert result == "test/file.txt"
        storage_service._mock_client.put_object.assert_called_once()

    def test_upload_bytes(self, storage_service):
        """Test bytes upload."""
        data = b"test content"

        result = storage_service.upload_bytes(
            bucket="models",
            object_name="test/file.txt",
            data=data,
            content_type="text/plain",
        )

        assert result == "test/file.txt"

    def test_download_file(self, storage_service):
        """Test file download."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"downloaded content"
        storage_service._mock_client.get_object.return_value = mock_response

        result = storage_service.download_file(
            bucket="models",
            object_name="test/file.txt",
        )

        assert result == b"downloaded content"
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()

    def test_download_range(self, storage_service):
        """Test byte range download."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"partial"
        storage_service._mock_client.get_object.return_value = mock_response

        result = storage_service.download_range(
            bucket="models",
            object_name="test/file.bin",
            offset=100,
            length=7,
        )

        assert result == b"partial"
        storage_service._mock_client.get_object.assert_called_once_with(
            "models",
            "test/file.bin",
            offset=100,
            length=7,
        )

    def test_list_objects(self, storage_service):
        """Test object listing."""
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "test/file1.txt"
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "test/file2.txt"

        storage_service._mock_client.list_objects.return_value = [mock_obj1, mock_obj2]

        result = storage_service.list_objects(
            bucket="models",
            prefix="test/",
            recursive=True,
        )

        assert result == ["test/file1.txt", "test/file2.txt"]

    def test_delete_object(self, storage_service):
        """Test object deletion."""
        storage_service.delete_object(
            bucket="models",
            object_name="test/file.txt",
        )

        storage_service._mock_client.remove_object.assert_called_once_with(
            "models",
            "test/file.txt",
        )

    def test_delete_prefix(self, storage_service):
        """Test prefix deletion."""
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "test/file1.txt"
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "test/file2.txt"

        storage_service._mock_client.list_objects.return_value = [mock_obj1, mock_obj2]

        storage_service.delete_prefix(
            bucket="models",
            prefix="test/",
        )

        assert storage_service._mock_client.remove_object.call_count == 2

    def test_object_exists_true(self, storage_service):
        """Test object exists check when object exists."""
        storage_service._mock_client.stat_object.return_value = MagicMock()

        result = storage_service.object_exists(
            bucket="models",
            object_name="test/file.txt",
        )

        assert result is True

    def test_object_exists_false(self, storage_service):
        """Test object exists check when object doesn't exist."""
        from minio.error import S3Error
        storage_service._mock_client.stat_object.side_effect = S3Error(
            "NoSuchKey", "NoSuchKey", None, None, None, None
        )

        result = storage_service.object_exists(
            bucket="models",
            object_name="nonexistent.txt",
        )

        assert result is False

    def test_get_object_size(self, storage_service):
        """Test getting object size."""
        mock_stat = MagicMock()
        mock_stat.size = 1024
        storage_service._mock_client.stat_object.return_value = mock_stat

        result = storage_service.get_object_size(
            bucket="models",
            object_name="test/file.bin",
        )

        assert result == 1024

    def test_get_presigned_url(self, storage_service):
        """Test presigned URL generation."""
        storage_service._mock_client.presigned_get_object.return_value = (
            "http://localhost:9000/models/test/file.txt?X-Amz-Signature=..."
        )

        result = storage_service.get_presigned_url(
            bucket="models",
            object_name="test/file.txt",
            expires_seconds=3600,
        )

        assert "localhost:9000" in result
        assert "test/file.txt" in result

    def test_is_available_true(self, storage_service):
        """Test availability check when MinIO is available."""
        storage_service._mock_client.list_buckets.return_value = []

        result = storage_service.is_available()

        assert result is True

    def test_is_available_false(self, storage_service):
        """Test availability check when MinIO is unavailable."""
        storage_service._mock_client.list_buckets.side_effect = Exception("Connection refused")

        result = storage_service.is_available()

        assert result is False


class TestStorageServiceEnsureReady:
    """Tests for _ensure_ready behavior."""

    def test_ensure_ready_raises_when_unavailable(self):
        """Test that operations fail when storage is not available."""
        with patch("app.services.storage.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                minio_endpoint="localhost:9000",
                minio_access_key="test",
                minio_secret_key="test",
                minio_secure=False,
                minio_bucket_models="models",
                minio_bucket_results="results",
            )
            with patch("app.services.storage.Minio") as mock_minio:
                mock_client = MagicMock()
                # Fail all bucket existence checks
                mock_client.bucket_exists.side_effect = Exception("Connection refused")
                mock_minio.return_value = mock_client

                service = StorageService(max_retries=1, retry_delay=0.01)

                with pytest.raises(RuntimeError, match="MinIO storage is not available"):
                    service.upload_bytes("models", "test.txt", b"data")


class TestStorageServiceSingleton:
    """Tests for storage service singleton management."""

    def test_get_storage_service_creates_singleton(self):
        """Test that get_storage_service creates a singleton."""
        reset_storage_service()

        with patch("app.services.storage.StorageService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            service1 = get_storage_service()
            service2 = get_storage_service()

            assert service1 is service2
            mock_cls.assert_called_once()

        reset_storage_service()

    def test_reset_storage_service_clears_singleton(self):
        """Test that reset_storage_service clears the singleton."""
        reset_storage_service()

        with patch("app.services.storage.StorageService") as mock_cls:
            mock_instance1 = MagicMock()
            mock_instance2 = MagicMock()
            mock_cls.side_effect = [mock_instance1, mock_instance2]

            service1 = get_storage_service()
            reset_storage_service()
            service2 = get_storage_service()

            assert service1 is not service2
            assert mock_cls.call_count == 2

        reset_storage_service()


class TestWaitForStorage:
    """Tests for wait_for_storage function."""

    def test_wait_for_storage_success(self):
        """Test waiting for storage when it becomes available."""
        reset_storage_service()

        with patch("app.services.storage.StorageService") as mock_cls:
            mock_service = MagicMock()
            mock_service.is_available.return_value = True
            mock_cls.return_value = mock_service

            result = wait_for_storage(max_retries=1, retry_delay=0.01)

            assert result is True

        reset_storage_service()

    def test_wait_for_storage_timeout(self):
        """Test waiting for storage when it never becomes available."""
        reset_storage_service()

        with patch("app.services.storage.StorageService") as mock_cls:
            mock_service = MagicMock()
            mock_service.is_available.return_value = False
            mock_cls.return_value = mock_service

            result = wait_for_storage(max_retries=2, retry_delay=0.01)

            assert result is False

        reset_storage_service()
