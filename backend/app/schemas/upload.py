"""Upload job schemas for tracking multi-stage upload progress."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class UploadStage(str, Enum):
    """Stages of the upload process."""
    RECEIVING = "receiving"      # File being uploaded to server
    EXTRACTING = "extracting"    # Extracting ZIP contents
    VALIDATING = "validating"    # Running FloPy validation
    STORING = "storing"          # Uploading files to MinIO
    CACHING = "caching"          # Generating grid/array caches
    COMPLETE = "complete"        # All done
    FAILED = "failed"            # Error occurred


class UploadStatus(BaseModel):
    """Current status of an upload job."""
    job_id: str
    project_id: str
    stage: UploadStage
    progress: int = 0  # 0-100 within current stage
    message: str = ""
    file_count: Optional[int] = None
    files_processed: Optional[int] = None
    is_valid: Optional[bool] = None
    error: Optional[str] = None
