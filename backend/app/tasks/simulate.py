"""Simulation task definitions."""

import asyncio
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

from celery import current_task
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.base import SessionLocal
from app.models.project import Project, Run, RunStatus
from app.services.path_normalizer import normalize_all_model_files
from app.services.redis_manager import get_sync_client
from app.services.storage import get_storage_service
from celery_app import celery_app

settings = get_settings()


def detect_modflow_executable(model_type: str) -> Optional[str]:
    """Detect the appropriate MODFLOW executable based on model type."""
    exe_map = {
        "mf6": [settings.mf6_exe_path, "mf6", "/usr/local/bin/mf6"],
        "mf2005": [settings.mf2005_exe_path, "mf2005", "/usr/local/bin/mf2005"],
        "mfnwt": [settings.mfnwt_exe_path, "mfnwt", "/usr/local/bin/mfnwt"],
        "mfusg": [settings.mfusg_exe_path, "mfusg", "/usr/local/bin/mfusg"],
    }

    paths = exe_map.get(model_type, [])
    for exe_path in paths:
        if exe_path and shutil.which(exe_path):
            return exe_path

    return None


def detect_model_type_from_dir(model_dir: Path) -> str:
    """Detect MODFLOW model type from directory contents."""
    files = [f.name.lower() for f in model_dir.iterdir() if f.is_file()]

    # MODFLOW 6
    if "mfsim.nam" in files:
        return "mf6"

    # Check NAM files for USG/NWT indicators
    nam_files = [f for f in model_dir.iterdir() if f.suffix.lower() == ".nam"]
    for nam_file in nam_files:
        try:
            content = nam_file.read_text().upper()
            if "DISU" in content or "CLN" in content or "GNC" in content:
                return "mfusg"
            if "UPW" in content or "NWT" in content:
                return "mfnwt"
        except Exception:
            pass

    # Default to mf2005
    if nam_files:
        return "mf2005"

    return "mf6"  # Default fallback


def get_model_nam_file(model_dir: Path, model_type: str) -> Optional[str]:
    """Get the name file for the model."""
    if model_type == "mf6":
        if (model_dir / "mfsim.nam").exists():
            return "mfsim.nam"
    else:
        for f in model_dir.iterdir():
            if f.suffix.lower() == ".nam" and f.name.lower() != "mfsim.nam":
                return f.name
    return None


def enable_budget_output(model_dir: Path, publish) -> bool:
    """
    Modify OC file to enable cell-by-cell budget output.

    Finds the OC file referenced in the NAM file and adds SAVE BUDGET
    statements to each stress period if not already present.

    Returns True if modification was successful.
    """
    import re

    # Find NAM file
    nam_files = list(model_dir.glob("*.nam"))
    nam_files = [f for f in nam_files if f.name.lower() != "mfsim.nam"]

    if not nam_files:
        publish("Note: No NAM file found, cannot enable budget output")
        return False

    nam_file = nam_files[0]
    nam_content = nam_file.read_text()

    # Find OC file reference in NAM
    oc_match = re.search(r'^\s*OC\s+\d+\s+(\S+)', nam_content, re.MULTILINE | re.IGNORECASE)
    if not oc_match:
        publish("Note: No OC file referenced in NAM, cannot enable budget output")
        return False

    oc_filename = oc_match.group(1)
    oc_file = model_dir / oc_filename

    if not oc_file.exists():
        publish(f"Note: OC file {oc_filename} not found")
        return False

    oc_content = oc_file.read_text()

    # Check if SAVE BUDGET is already present
    if re.search(r'SAVE\s+BUDGET', oc_content, re.IGNORECASE):
        publish("Budget output already enabled in OC file")
        return True

    # Add SAVE BUDGET after each SAVE HEAD statement
    modified_content = re.sub(
        r'(SAVE\s+HEAD\b)',
        r'\1\n     SAVE BUDGET',
        oc_content,
        flags=re.IGNORECASE
    )

    if modified_content == oc_content:
        # No SAVE HEAD found, try adding after each PERIOD line
        modified_content = re.sub(
            r'(PERIOD\s+\d+\s+STEP\s+\d+)',
            r'\1\n     SAVE BUDGET',
            oc_content,
            flags=re.IGNORECASE
        )

    if modified_content != oc_content:
        oc_file.write_text(modified_content)
        publish(f"Enabled budget output in {oc_filename}")
        return True

    publish("Note: Could not modify OC file to enable budget output")
    return False


def enable_mf6_budget_output(model_dir: Path, publish, save_cbc: bool = True) -> bool:
    """
    Modify MF6 Output Control file to enable budget output.

    Parses mfsim.nam → finds model NAM → finds OC6 filename → modifies it.

    When save_cbc=True: adds BUDGET FILEOUT, SAVE BUDGET, and PRINT BUDGET.
    When save_cbc=False: only adds PRINT BUDGET to PERIOD blocks so the
    volumetric budget summary (with PERCENT DISCREPANCY) appears in the
    listing file for mass balance error parsing.

    Returns True if modification was successful.
    """
    import re

    # Step 1: Parse mfsim.nam to find model NAM filename
    mfsim_path = model_dir / "mfsim.nam"
    if not mfsim_path.exists():
        publish("Note: mfsim.nam not found, cannot enable MF6 budget output")
        return False

    mfsim_content = mfsim_path.read_text()

    # Find model NAM in MODELS block (format: type6 modelname.nam modelname)
    model_nam = None
    in_models = False
    for line in mfsim_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.upper().startswith("BEGIN MODELS"):
            in_models = True
            continue
        if stripped.upper().startswith("END MODELS"):
            break
        if in_models:
            parts = stripped.split()
            if len(parts) >= 2:
                model_nam = parts[1]
                break

    if not model_nam:
        publish("Note: Could not find model NAM in mfsim.nam MODELS block")
        return False

    # Step 2: Parse model NAM to find OC6 filename
    model_nam_path = model_dir / model_nam
    if not model_nam_path.exists():
        publish(f"Note: Model NAM file {model_nam} not found")
        return False

    model_nam_content = model_nam_path.read_text()

    oc_filename = None
    in_packages = False
    for line in model_nam_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.upper().startswith("BEGIN PACKAGES"):
            in_packages = True
            continue
        if stripped.upper().startswith("END PACKAGES"):
            break
        if in_packages:
            parts = stripped.split()
            if len(parts) >= 2:
                # MF6 NAM PACKAGES format: ftype  fname  [pname]
                pkg_type = parts[0].upper()
                pkg_file = parts[1]
                # OC can appear as type or the file may contain "oc" in extension
                if pkg_type in ("OC6", "OC") or pkg_file.lower().endswith(".oc"):
                    oc_filename = pkg_file
                    break

    if not oc_filename:
        publish("Note: No OC package found in model NAM, cannot enable MF6 budget output")
        return False

    # Step 3: Modify OC file
    oc_path = model_dir / oc_filename
    if not oc_path.exists():
        publish(f"Note: OC file {oc_filename} not found")
        return False

    oc_content = oc_path.read_text()
    modified = False

    # Derive basename for budget file from model NAM
    basename = Path(model_nam).stem

    # Add BUDGET FILEOUT to OPTIONS block if not present (only when saving CBC)
    if save_cbc and not re.search(r'BUDGET\s+FILEOUT', oc_content, re.IGNORECASE):
        budget_line = f"  BUDGET FILEOUT {basename}.cbc"
        # Insert after BEGIN OPTIONS or after existing FILEOUT lines
        if re.search(r'BEGIN\s+OPTIONS', oc_content, re.IGNORECASE):
            oc_content = re.sub(
                r'(BEGIN\s+OPTIONS[^\n]*\n)',
                r'\1' + budget_line + '\n',
                oc_content,
                count=1,
                flags=re.IGNORECASE,
            )
            modified = True
        else:
            # No OPTIONS block - add one before the first PERIOD block
            options_block = f"BEGIN OPTIONS\n{budget_line}\nEND OPTIONS\n\n"
            period_match = re.search(r'BEGIN\s+PERIOD', oc_content, re.IGNORECASE)
            if period_match:
                oc_content = oc_content[:period_match.start()] + options_block + oc_content[period_match.start():]
                modified = True

    # Add SAVE BUDGET and PRINT BUDGET to each PERIOD block if not present
    # PRINT BUDGET ensures the volumetric budget (with PERCENT DISCREPANCY)
    # is written to the listing file for convergence parsing.
    period_blocks = list(re.finditer(
        r'(BEGIN\s+PERIOD\s+\d+)(.*?)(END\s+PERIOD)',
        oc_content,
        re.IGNORECASE | re.DOTALL,
    ))

    if period_blocks:
        # Process in reverse to preserve positions
        for match in reversed(period_blocks):
            block_content = match.group(2)
            additions = []
            if save_cbc and not re.search(r'SAVE\s+BUDGET', block_content, re.IGNORECASE):
                additions.append('SAVE BUDGET')
            if not re.search(r'PRINT\s+BUDGET', block_content, re.IGNORECASE):
                additions.append('PRINT BUDGET')

            if additions:
                # Match the pattern of SAVE HEAD if present for indentation
                head_save = re.search(r'(\s+SAVE\s+HEAD\b[^\n]*)', block_content, re.IGNORECASE)
                if head_save:
                    insert_pos = match.start(2) + head_save.end()
                    indent = re.match(r'\s*', head_save.group(1)).group(0)
                    insert_text = ''.join(f"\n{indent}{a}" for a in additions)
                    oc_content = (
                        oc_content[:insert_pos]
                        + insert_text
                        + oc_content[insert_pos:]
                    )
                else:
                    # Insert at the end of the period block
                    insert_pos = match.start(3)
                    insert_text = ''.join(f"  {a} LAST\n" for a in additions)
                    oc_content = (
                        oc_content[:insert_pos]
                        + insert_text
                        + oc_content[insert_pos:]
                    )
                modified = True

    if modified:
        oc_path.write_text(oc_content)
        if save_cbc:
            publish(f"Enabled MF6 budget output in {oc_filename}")
        else:
            publish(f"Enabled budget printing in listing file for {oc_filename}")
        return True

    publish("MF6 budget output already enabled in OC file")
    return True


@celery_app.task(bind=True, name="app.tasks.simulate.run_forward_model")
def run_forward_model(self, run_id: str, project_id: str, save_budget: bool = False) -> dict:
    """
    Execute a forward MODFLOW simulation.

    This task:
    1. Copies model files from MinIO to a working directory
    2. Optionally modifies OC file to enable budget output
    3. Executes MODFLOW as a subprocess
    4. Streams stdout to Redis pub/sub for live console
    5. Updates run status in the database
    6. Uploads results back to MinIO

    Args:
        run_id: UUID of the run record
        project_id: UUID of the project
        save_budget: If True, modify OC file to enable CBC output

    Returns:
        Dict with run results and status
    """
    redis_client = get_sync_client()
    channel = f"simulation:{run_id}:output"
    history_key = f"simulation:{run_id}:history"
    storage = get_storage_service()

    def publish(message: str):
        """Publish message to Redis channel and append to history list."""
        try:
            redis_client.publish(channel, message)
            redis_client.rpush(history_key, message)
            redis_client.ltrim(history_key, -20000, -1)
            # Set TTL on history (24 hours) — refreshed on each append
            redis_client.expire(history_key, 86400)
        except Exception as e:
            logger.warning(f"Redis publish error (simulation continues): {e}")

    with SessionLocal() as db:
        # Get project and run
        project = db.execute(
            select(Project).where(Project.id == UUID(project_id))
        ).scalar_one_or_none()

        run = db.execute(
            select(Run).where(Run.id == UUID(run_id))
        ).scalar_one_or_none()

        if not project or not run:
            return {"error": "Project or run not found"}

        # Idempotency guard: if the task was requeued (e.g. worker restart),
        # don't re-run a simulation that already completed or is still running
        if run.status == RunStatus.COMPLETED:
            publish("Task requeued but simulation already completed — skipping.")
            return {"status": "already_completed", "run_id": run_id}
        if run.status == RunStatus.FAILED:
            publish("Task requeued but simulation already failed — skipping.")
            return {"status": "already_failed", "run_id": run_id}
        if run.status == RunStatus.RUNNING:
            # Another worker may already be executing this task.
            # Use a Redis lock to prevent duplicate execution.
            lock_key = f"simulation_lock:{run_id}"
            acquired = redis_client.set(lock_key, "1", nx=True, ex=7200)  # 2hr TTL
            if not acquired:
                publish("Task requeued but simulation is already running — skipping duplicate.")
                return {"status": "already_running", "run_id": run_id}

        if not project.storage_path:
            run.status = RunStatus.FAILED
            run.error_message = "No model files found"
            db.commit()
            return {"error": "No model files found"}

        model_type = project.model_type.value if project.model_type else "mf6"

        # Update run status to running
        run.status = RunStatus.RUNNING
        run.started_at = datetime.utcnow()
        run.celery_task_id = self.request.id
        db.commit()

        # Set the execution lock (for tasks that start in QUEUED/PENDING status)
        lock_key = f"simulation_lock:{run_id}"
        redis_client.set(lock_key, "1", nx=True, ex=7200)

        publish(f"Starting simulation (model type: {model_type})...")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()

            # Download model files
            try:
                publish("Downloading model files...")
                files = storage.list_objects(
                    settings.minio_bucket_models,
                    prefix=project.storage_path,
                    recursive=True,
                )

                file_count = 0
                for obj_name in files:
                    rel_path = obj_name[len(project.storage_path):].lstrip("/")
                    if not rel_path:
                        continue

                    local_path = model_dir / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    storage.download_to_file(settings.minio_bucket_models, obj_name, local_path)
                    file_count += 1

                publish(f"Downloaded {file_count} files")

                # Normalize backslash paths in model files (Windows → Linux)
                files_fixed, paths_fixed = normalize_all_model_files(model_dir)
                if paths_fixed > 0:
                    publish(f"Normalized {paths_fixed} path(s) in {files_fixed} file(s)")

                # Auto-detect model type if unknown
                if model_type == "unknown":
                    model_type = detect_model_type_from_dir(model_dir)
                    publish(f"Auto-detected model type: {model_type}")

                # Always ensure MF6 has PRINT BUDGET in OC for listing file mass balance.
                # Only add SAVE BUDGET + BUDGET FILEOUT when user requests CBC output.
                if model_type == "mf6":
                    enable_mf6_budget_output(model_dir, publish, save_cbc=save_budget)
                elif save_budget:
                    enable_budget_output(model_dir, publish)

            except Exception as e:
                run.status = RunStatus.FAILED
                run.error_message = f"Failed to download model files: {str(e)}"
                run.completed_at = datetime.utcnow()
                db.commit()
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Find executable (may need to re-detect after auto-detection)
            executable = detect_modflow_executable(model_type)
            if not executable:
                run.status = RunStatus.FAILED
                run.error_message = f"MODFLOW executable not found for {model_type}"
                run.completed_at = datetime.utcnow()
                db.commit()
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            publish(f"Using {model_type.upper()} executable: {executable}")

            # Get nam file
            nam_file = get_model_nam_file(model_dir, model_type)
            if not nam_file:
                run.status = RunStatus.FAILED
                run.error_message = "Could not find model name file"
                run.completed_at = datetime.utcnow()
                db.commit()
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Build command
            if model_type == "mf6":
                cmd = [executable]
            else:
                cmd = [executable, nam_file]

            publish("-" * 60)
            publish(f"Running: {' '.join(cmd)}")
            publish("-" * 60)

            # Start live result processing in a background thread (not Celery task,
            # since concurrency=1 means a queued task can't run during simulation)
            live_thread = None
            try:
                from app.tasks.live_results import live_results_thread_fn
                live_thread = threading.Thread(
                    target=live_results_thread_fn,
                    args=(run_id, project_id, str(model_dir), model_type),
                    daemon=True,
                )
                live_thread.start()
                publish("Live result processing started...")
            except Exception as e:
                publish(f"Note: Live result processing not available: {e}")

            # Run simulation
            process = None
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(model_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Use a reader thread to drain stdout so the pipe never
                # blocks MODFLOW while we do batched Redis I/O.
                def _stdout_reader(pipe, q):
                    try:
                        for raw_line in iter(pipe.readline, ""):
                            q.put(raw_line.rstrip())
                    finally:
                        q.put(None)  # sentinel

                stdout_q: queue.Queue = queue.Queue(maxsize=10000)
                reader_thread = threading.Thread(
                    target=_stdout_reader,
                    args=(process.stdout, stdout_q),
                    daemon=True,
                )
                reader_thread.start()

                output_lines = []
                cancel_check_counter = 0
                eof = False
                while not eof:
                    # Drain available lines (up to 100 per batch)
                    batch = []
                    try:
                        while len(batch) < 100:
                            line = stdout_q.get(timeout=0.1)
                            if line is None:
                                eof = True
                                break
                            batch.append(line)
                    except queue.Empty:
                        pass

                    if batch:
                        output_lines.extend(batch)
                        # Batched Redis operations — single round-trip
                        try:
                            pipe = redis_client.pipeline(transaction=False)
                            for b_line in batch:
                                pipe.publish(channel, b_line)
                                pipe.rpush(history_key, b_line)
                            pipe.ltrim(history_key, -20000, -1)
                            pipe.expire(history_key, 86400)
                            pipe.execute()
                        except Exception as e:
                            logger.warning(f"Redis pipeline error (simulation continues): {e}")

                    # Check cancellation every ~500 lines instead of every line
                    cancel_check_counter += len(batch)
                    if cancel_check_counter >= 500:
                        cancel_check_counter = 0
                        if redis_client.exists(f"cancel:{run_id}"):
                            process.terminate()
                            process.wait()
                            if live_thread is not None:
                                live_thread.join(timeout=10)
                            run.status = RunStatus.CANCELLED
                            run.error_message = "Cancelled by user"
                            run.completed_at = datetime.utcnow()
                            db.commit()
                            publish("Simulation cancelled by user")
                            publish("__STATUS__:cancelled")
                            redis_client.delete(f"cancel:{run_id}")
                            return {"status": "cancelled"}

                return_code = process.wait()

                # Signal live results thread to stop and wait for it
                if live_thread is not None:
                    live_thread.join(timeout=10)

                publish("-" * 60)

                if return_code == 0:
                    run.status = RunStatus.COMPLETED
                    run.exit_code = return_code
                    publish("Simulation completed successfully!")
                else:
                    run.status = RunStatus.FAILED
                    run.exit_code = return_code
                    run.error_message = f"Simulation failed with return code {return_code}"
                    publish(f"Simulation failed with return code {return_code}")

                run.completed_at = datetime.utcnow()

                # Commit status immediately so it's saved even if upload phase crashes (OOM)
                db.commit()

                # Collect and upload output files
                # Include common MODFLOW output extensions:
                # - .hds/.hed: binary head files
                # - .cbc/.bud: cell budget files
                # - .ddn: drawdown files
                # - .ucn: concentration files (MT3D)
                # - .lst/.list: listing files
                # - .grb: binary grid file (MF6)
                output_extensions = {
                    ".hds", ".hed",  # Head files
                    ".cbc", ".bud", ".cbb",  # Budget files
                    ".ddn",          # Drawdown
                    ".ucn",          # Concentrations
                    ".lst", ".list", # Listing files
                    ".grb",          # Grid file
                }
                output_files = []

                if run.status == RunStatus.COMPLETED:
                    output_prefix = f"{project.storage_path}/output/{run_id}"
                    for f in model_dir.rglob("*"):
                        if f.is_file() and f.suffix.lower() in output_extensions:
                            obj_name = f"{output_prefix}/{f.name}"
                            try:
                                # Stream from disk instead of reading entire file into memory
                                file_size = f.stat().st_size
                                with open(f, "rb") as fh:
                                    storage.upload_file(
                                        settings.minio_bucket_models,
                                        obj_name,
                                        fh,
                                        file_size,
                                    )
                                output_files.append(f.name)
                            except Exception as e:
                                publish(f"Warning: Failed to upload {f.name}: {e}")

                    if output_files:
                        publish(f"Uploaded {len(output_files)} output files")
                        run.results_path = output_prefix

                db.commit()

                # Auto-trigger post-processing for completed runs
                if run.status == RunStatus.COMPLETED and run.results_path:
                    try:
                        from app.tasks.postprocess import extract_results
                        # Use quick_mode=True for fast post-processing
                        # Head slices are fetched on-demand when requested
                        extract_results.delay(str(run.id), str(project_id), True)
                        publish("Post-processing queued (quick mode)...")
                    except Exception as e:
                        # Record the failure so dashboard knows post-processing didn't start
                        publish(f"Warning: Failed to queue post-processing: {e}")
                        ci = run.convergence_info or {}
                        ci["postprocess_error"] = str(e)
                        run.convergence_info = ci
                        db.commit()

                status = "completed" if run.status == RunStatus.COMPLETED else "failed"
                publish(f"__STATUS__:{status}")

                # Release execution lock
                redis_client.delete(f"simulation_lock:{run_id}")

                return {
                    "run_id": run_id,
                    "status": status,
                    "return_code": return_code,
                    "output_files": output_files,
                }

            except Exception as e:
                if process is not None:
                    process.terminate()
                    process.wait()
                run.status = RunStatus.FAILED
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
                db.commit()
                publish(f"ERROR: {str(e)}")
                publish("__STATUS__:failed")
                # Release execution lock
                redis_client.delete(f"simulation_lock:{run_id}")
                return {"error": str(e)}
            finally:
                if process is not None and process.stdout:
                    try:
                        process.stdout.close()
                    except Exception:
                        pass


@celery_app.task(bind=True, name="app.tasks.simulate.cancel_run")
def cancel_run(self, run_id: str) -> dict:
    """Cancel a running simulation."""
    with SessionLocal() as db:
        run = db.execute(
            select(Run).where(Run.id == UUID(run_id))
        ).scalar_one_or_none()

        if not run:
            return {"error": "Run not found"}

        # Set cancellation flag in Redis
        redis_client = get_sync_client()
        redis_client.setex(f"cancel:{run_id}", 300, "1")  # Expires in 5 minutes

        if run.celery_task_id:
            # Revoke the Celery task
            celery_app.control.revoke(run.celery_task_id, terminate=True)

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.utcnow()
        run.error_message = "Cancelled by user"
        db.commit()

        # Notify via Redis pub/sub
        redis_client.publish(f"simulation:{run_id}:output", "Simulation cancelled")
        redis_client.publish(f"simulation:{run_id}:output", "__STATUS__:cancelled")

        return {
            "run_id": run_id,
            "status": "cancelled",
        }
