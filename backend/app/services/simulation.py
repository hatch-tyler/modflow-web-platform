"""MODFLOW simulation execution service."""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional
from uuid import UUID

from app.config import get_settings
from app.services.storage import get_storage_service

settings = get_settings()


class SimulationStatus(str, Enum):
    """Simulation status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    status: SimulationStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None
    output_lines: list = field(default_factory=list)
    error_message: Optional[str] = None
    output_files: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "return_code": self.return_code,
            "output_line_count": len(self.output_lines),
            "error_message": self.error_message,
            "output_files": self.output_files,
        }


def detect_modflow_executable(model_type: str) -> Optional[str]:
    """
    Detect the appropriate MODFLOW executable based on model type.

    Returns path to executable or None if not found.
    """
    exe_map = {
        "mf6": [settings.mf6_exe_path, "mf6", "/usr/local/bin/mf6"],
        "mf2005": [settings.mf2005_exe_path, "mf2005", "/usr/local/bin/mf2005"],
        "mfnwt": [settings.mfnwt_exe_path, "mfnwt", "/usr/local/bin/mfnwt"],
        "mfusg": ["/usr/local/bin/mfusg", "mfusg"],
    }

    # Get list of possible paths for this model type
    paths = exe_map.get(model_type, [])

    for exe_path in paths:
        if exe_path and shutil.which(exe_path):
            return exe_path

    return None


def get_model_nam_file(model_dir: Path, model_type: str) -> Optional[str]:
    """
    Get the name file for the model.

    Returns the filename (not full path) of the nam file.
    """
    if model_type == "mf6":
        # MF6 uses mfsim.nam
        if (model_dir / "mfsim.nam").exists():
            return "mfsim.nam"
    else:
        # MF2005/NWT/USG use .nam files
        for f in model_dir.iterdir():
            if f.suffix.lower() == ".nam" and f.name.lower() != "mfsim.nam":
                return f.name

    return None


async def run_simulation(
    project_id: UUID,
    storage_path: str,
    model_type: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> SimulationResult:
    """
    Run a MODFLOW simulation.

    Args:
        project_id: Project UUID
        storage_path: Path to model files in storage
        model_type: Type of MODFLOW model (mf6, mf2005, mfnwt, mfusg)
        on_output: Optional callback for each line of output

    Returns:
        SimulationResult with status and output
    """
    result = SimulationResult(status=SimulationStatus.PENDING)

    # Find executable
    executable = detect_modflow_executable(model_type)
    if not executable:
        result.status = SimulationStatus.FAILED
        result.error_message = f"MODFLOW executable not found for model type: {model_type}"
        return result

    # Download model files from storage
    storage = get_storage_service()

    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()

        # Download all model files
        try:
            files = storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_path,
                recursive=True,
            )

            for obj_name in files:
                rel_path = obj_name[len(storage_path) :].lstrip("/")
                if not rel_path:
                    continue

                local_path = model_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                file_data = storage.download_file(settings.minio_bucket_models, obj_name)
                local_path.write_bytes(file_data)

        except Exception as e:
            result.status = SimulationStatus.FAILED
            result.error_message = f"Failed to download model files: {str(e)}"
            return result

        # Get the nam file
        nam_file = get_model_nam_file(model_dir, model_type)
        if not nam_file:
            result.status = SimulationStatus.FAILED
            result.error_message = "Could not find model name file"
            return result

        # Build command
        if model_type == "mf6":
            # MF6 just needs to run in the model directory
            cmd = [executable]
        else:
            # MF2005/NWT/USG need the nam file as argument
            cmd = [executable, nam_file]

        # Run simulation
        result.status = SimulationStatus.RUNNING
        result.started_at = datetime.utcnow()

        if on_output:
            on_output(f"Starting {model_type.upper()} simulation...")
            on_output(f"Executable: {executable}")
            on_output(f"Working directory: {model_dir}")
            on_output("-" * 60)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(model_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").rstrip()
                result.output_lines.append(line_str)

                if on_output:
                    on_output(line_str)

            # Wait for completion
            return_code = await process.wait()
            result.return_code = return_code
            result.finished_at = datetime.utcnow()

            if return_code == 0:
                result.status = SimulationStatus.COMPLETED
                if on_output:
                    on_output("-" * 60)
                    on_output("Simulation completed successfully!")
            else:
                result.status = SimulationStatus.FAILED
                result.error_message = f"Simulation failed with return code {return_code}"
                if on_output:
                    on_output("-" * 60)
                    on_output(f"Simulation failed with return code {return_code}")

            # Collect output files (supports common MODFLOW output extensions)
            output_extensions = {".hds", ".hed", ".cbc", ".bud", ".ddn", ".ucn", ".lst", ".list", ".grb"}
            for f in model_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in output_extensions:
                    result.output_files.append(f.name)

            # Upload output files back to storage
            if result.status == SimulationStatus.COMPLETED:
                output_prefix = f"{storage_path}/output"
                for f in model_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in output_extensions:
                        obj_name = f"{output_prefix}/{f.name}"
                        storage.upload_bytes(
                            settings.minio_bucket_models,
                            obj_name,
                            f.read_bytes(),
                        )

        except asyncio.CancelledError:
            result.status = SimulationStatus.CANCELLED
            result.finished_at = datetime.utcnow()
            result.error_message = "Simulation was cancelled"
            if on_output:
                on_output("Simulation cancelled by user")
            raise

        except Exception as e:
            result.status = SimulationStatus.FAILED
            result.finished_at = datetime.utcnow()
            result.error_message = str(e)
            if on_output:
                on_output(f"Error: {str(e)}")

    return result


async def stream_simulation_output(
    project_id: UUID,
    storage_path: str,
    model_type: str,
) -> AsyncGenerator[str, None]:
    """
    Run simulation and yield output lines as they are produced.

    This is useful for Server-Sent Events streaming.
    """
    output_queue: asyncio.Queue = asyncio.Queue()

    def on_output(line: str):
        output_queue.put_nowait(line)

    # Start simulation in background
    task = asyncio.create_task(
        run_simulation(project_id, storage_path, model_type, on_output)
    )

    try:
        while not task.done():
            try:
                # Wait for output with timeout
                line = await asyncio.wait_for(output_queue.get(), timeout=0.5)
                yield line
            except asyncio.TimeoutError:
                # No output, check if task is still running
                continue

        # Drain remaining output
        while not output_queue.empty():
            yield output_queue.get_nowait()

        # Get final result
        result = task.result()
        yield f"__STATUS__:{result.status.value}"

    except asyncio.CancelledError:
        task.cancel()
        yield "__STATUS__:cancelled"
        raise
