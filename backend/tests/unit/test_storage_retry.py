"""Tests for storage service retry logic.

Covers:
- Transient failures trigger retries with exponential backoff
- Client errors (4xx) are NOT retried
- Server errors (5xx) ARE retried
- Connection errors ARE retried
- Max attempts exhausted raises the last exception
"""

import io
from unittest.mock import MagicMock, patch

import pytest

from app.services.storage import StorageService, _retry_on_transient


def _make_storage_service():
    """Create a StorageService with mocked Minio client and settings."""
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
            return service, mock_client


class TestRetryOnTransientDecorator:
    """Test the _retry_on_transient decorator directly."""

    def test_succeeds_on_first_attempt(self):
        """If the function succeeds, it should be called once."""
        call_count = 0

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def good_fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = good_fn()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_connection_error(self):
        """ConnectionError should trigger retries."""
        call_count = 0

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return "ok"

        result = flaky_fn()
        assert result == "ok"
        assert call_count == 3

    def test_retries_on_os_error(self):
        """OSError should trigger retries."""
        call_count = 0

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Network unreachable")
            return "ok"

        result = flaky_fn()
        assert result == "ok"
        assert call_count == 2

    def test_retries_on_5xx_s3_error(self):
        """S3Error with code starting with '5' should trigger retries."""
        from minio.error import S3Error

        call_count = 0
        mock_resp = MagicMock()

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def server_error_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # The retry decorator checks str(e.code).startswith("5")
                raise S3Error("500", "Internal Server Error", None, None, None, mock_resp)
            return "ok"

        result = server_error_fn()
        assert result == "ok"
        assert call_count == 3

    def test_no_retry_on_s3_error_with_string_code(self):
        """S3Error with non-numeric code (e.g. 'NoSuchKey') is NOT retried."""
        from minio.error import S3Error

        call_count = 0
        mock_resp = MagicMock()

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def client_error_fn():
            nonlocal call_count
            call_count += 1
            raise S3Error("NoSuchKey", "NoSuchKey", None, None, None, mock_resp)

        with pytest.raises(S3Error):
            client_error_fn()

        assert call_count == 1  # No retries

    def test_no_retry_on_unknown_error(self):
        """Non-transient errors (like ValueError) should NOT be retried."""
        call_count = 0

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def bad_fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            bad_fn()

        assert call_count == 1

    def test_exhausted_retries_raises_last_exception(self):
        """After max_attempts, the last exception should be raised."""
        call_count = 0

        @_retry_on_transient(max_attempts=3, initial_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Attempt {call_count}")

        with pytest.raises(ConnectionError, match="Attempt 3"):
            always_fail()

        assert call_count == 3

    def test_exponential_backoff_caps_at_max_delay(self):
        """Delay should increase but cap at max_delay."""
        import time

        call_count = 0
        call_times = []

        @_retry_on_transient(
            max_attempts=4, initial_delay=0.01, max_delay=0.03
        )
        def timed_fail():
            nonlocal call_count
            call_count += 1
            call_times.append(time.monotonic())
            if call_count < 4:
                raise ConnectionError("fail")
            return "ok"

        result = timed_fail()
        assert result == "ok"
        assert call_count == 4

        # Check that delays exist between calls (not instant)
        for i in range(1, len(call_times)):
            gap = call_times[i] - call_times[i - 1]
            assert gap >= 0.005  # At least some delay


class TestStorageServiceRetryIntegration:
    """Test retry behavior on actual StorageService methods."""

    def test_upload_retries_on_connection_error(self):
        """upload_file should retry on ConnectionError."""
        service, mock_client = _make_storage_service()

        call_count = 0

        def flaky_put(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return MagicMock()  # Successful put returns result object

        mock_client.put_object.side_effect = flaky_put

        # Should succeed after retries
        result = service.upload_file(
            "models", "test.txt", io.BytesIO(b"data"), 4
        )
        assert result == "test.txt"
        assert call_count >= 2

    def test_download_retries_on_connection_error(self):
        """download_file should retry on ConnectionError."""
        service, mock_client = _make_storage_service()

        call_count = 0
        mock_response = MagicMock()
        mock_response.read.return_value = b"content"

        def flaky_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("timeout")
            return mock_response

        mock_client.get_object.side_effect = flaky_get

        result = service.download_file("models", "test.txt")
        assert result == b"content"
        assert call_count >= 2

    def test_download_no_retry_on_404(self):
        """download_file should NOT retry on NoSuchKey (404)."""
        from minio.error import S3Error

        service, mock_client = _make_storage_service()

        mock_resp = MagicMock()
        mock_client.get_object.side_effect = S3Error(
            "NoSuchKey", "NoSuchKey", None, None, None, mock_resp
        )

        with pytest.raises(S3Error):
            service.download_file("models", "nonexistent.txt")

        # Should be called only once (no retry)
        assert mock_client.get_object.call_count == 1

    def test_download_range_retries(self):
        """download_range should retry on transient errors."""
        service, mock_client = _make_storage_service()

        call_count = 0
        mock_response = MagicMock()
        mock_response.read.return_value = b"partial"

        def flaky_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Network unreachable")
            return mock_response

        mock_client.get_object.side_effect = flaky_get

        result = service.download_range("models", "test.bin", 0, 7)
        assert result == b"partial"
        assert call_count >= 2
