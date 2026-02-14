"""Calibration and parameter estimation task definitions."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.base import SessionLocal
from app.models.project import Project, Run, RunStatus
from app.services.redis_manager import get_sync_client
from app.services.storage import get_storage_service
from celery_app import celery_app

settings = get_settings()


def _run_pest_parallel(
    pst_path: Path,
    workspace_dir: Path,
    num_workers: int,
    pestpp_exe: str,
    publish_fn,
    redis_client,
    run_id: str,
) -> int:
    """
    Run PEST++ with parallel workers using pyEMU.

    This uses pyemu.os_utils.start_workers() to spawn local parallel agents
    that connect to the PEST++ manager process.

    Args:
        pst_path: Path to the PST control file
        workspace_dir: The PEST workspace directory (will be master)
        num_workers: Number of parallel workers to spawn
        pestpp_exe: Path to the PEST++ executable (pestpp-glm or pestpp-ies)
        publish_fn: Function to publish messages to Redis
        redis_client: Redis client for cancellation checks
        run_id: Run ID for cancellation key

    Returns:
        Return code from PEST++ (0 = success)
    """
    import pyemu

    publish_fn(f"Starting PEST++ with {num_workers} parallel workers...")

    # Create worker template directory
    # The template dir must be a copy of the workspace with model files
    template_dir = workspace_dir / "worker_template"
    if template_dir.exists():
        shutil.rmtree(template_dir)
    shutil.copytree(workspace_dir, template_dir)

    # Create master directory
    master_dir = workspace_dir / "master"
    master_dir.mkdir(exist_ok=True)

    # Copy PST file and related files to master
    for f in workspace_dir.iterdir():
        if f.is_file() and f.name != "worker_template":
            shutil.copy2(f, master_dir / f.name)

    pst_name = pst_path.name

    # Determine the model executable from forward_run.py
    # The forward_run.py in the workspace handles running the model
    exe_name = "python"  # We use python forward_run.py

    # Thread to monitor for cancellation
    cancel_flag = threading.Event()

    def check_cancel():
        while not cancel_flag.is_set():
            if redis_client.exists(f"cancel:{run_id}"):
                cancel_flag.set()
                break
            cancel_flag.wait(2.0)

    cancel_thread = threading.Thread(target=check_cancel, daemon=True)
    cancel_thread.start()

    return_code = -1

    try:
        publish_fn(f"Template dir: {template_dir}")
        publish_fn(f"Master dir: {master_dir}")
        publish_fn(f"PST file: {pst_name}")
        publish_fn("-" * 60)

        # Use pyemu to start workers
        # This is a blocking call that runs until PEST++ completes
        pyemu.os_utils.start_workers(
            worker_dir=str(template_dir),
            exe_rel_path=exe_name,
            pst_rel_path=pst_name,
            num_workers=num_workers,
            master_dir=str(master_dir),
            worker_root=str(workspace_dir / "workers"),
            port=4004,
            silent_master=False,
        )

        # Check if the run was cancelled
        if cancel_flag.is_set():
            publish_fn("PEST++ run cancelled by user")
            return -1

        # Check for successful completion by looking for output files
        rec_file = master_dir / pst_name.replace(".pst", ".rec")
        if rec_file.exists():
            publish_fn("PEST++ completed successfully")
            return_code = 0
        else:
            publish_fn("PEST++ may have failed - no .rec file found")
            return_code = 1

    except Exception as e:
        publish_fn(f"Error during parallel PEST++ execution: {e}")
        return_code = 1
    finally:
        cancel_flag.set()
        cancel_thread.join(timeout=1.0)

    # Copy results back from master to workspace for parsing
    for f in master_dir.iterdir():
        if f.is_file():
            dest = workspace_dir / f.name
            if not dest.exists() or f.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(f, dest)

    return return_code


def _run_pest_sequential(
    pst_path: Path,
    workspace_dir: Path,
    pestpp_exe: str,
    publish_fn,
    redis_client,
    run_id: str,
) -> int:
    """
    Run PEST++ sequentially (single worker, original behavior).

    This is used when num_workers=1 or as a fallback.
    """
    cmd = [pestpp_exe, pst_path.name]
    publish_fn("-" * 60)
    publish_fn(f"Running: {' '.join(cmd)}")
    publish_fn(f"Working directory: {workspace_dir}")
    publish_fn("-" * 60)

    process = subprocess.Popen(
        cmd,
        cwd=str(workspace_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            publish_fn(line)

            # Check for cancellation
            if redis_client.exists(f"cancel:{run_id}"):
                process.terminate()
                process.wait()
                return -1

        return process.wait()
    except Exception:
        process.terminate()
        process.wait()
        raise
    finally:
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass


def _copy_workspace_to_shared_volume(
    workspace_dir: Path,
    publish_fn,
) -> Path:
    """
    Copy PEST workspace to shared volume for local container agents.

    Returns the path in the shared volume.
    """
    shared_workspace = Path(settings.pest_workspace_path)
    shared_workspace.mkdir(parents=True, exist_ok=True)

    # Clear any existing workspace
    for item in shared_workspace.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Copy workspace files
    publish_fn(f"Copying workspace to shared volume: {shared_workspace}")
    file_count = 0
    for f in workspace_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(workspace_dir)
            dest = shared_workspace / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            file_count += 1

    publish_fn(f"Copied {file_count} files to shared volume")
    return shared_workspace


def _start_local_agent_containers(
    num_agents: int,
    publish_fn,
    redis_client,
    run_id: str,
) -> bool:
    """
    Start local PEST++ agent containers via docker compose.

    Args:
        num_agents: Number of local agent containers to start
        publish_fn: Function to publish messages
        redis_client: Redis client for tracking
        run_id: Run ID for tracking

    Returns:
        True if containers started successfully
    """
    publish_fn(f"Starting {num_agents} local agent containers...")

    try:
        # Set environment variables for docker compose
        env = os.environ.copy()
        env["LOCAL_AGENTS"] = str(num_agents)
        env["AGENT_MEMORY_LIMIT"] = settings.pest_agent_memory_limit
        env["AGENT_CPU_LIMIT"] = str(settings.pest_agent_cpu_limit)
        env["AGENT_MEMORY_RESERVATION"] = settings.pest_agent_memory_reservation
        env["AGENT_CPU_RESERVATION"] = str(settings.pest_agent_cpu_reservation)

        # Find docker-compose.pest.yml
        # It should be in the project root (parent of backend)
        compose_file = Path("/app").parent / settings.pest_compose_file
        if not compose_file.exists():
            # Try current directory
            compose_file = Path(settings.pest_compose_file)

        if not compose_file.exists():
            publish_fn(f"Warning: {settings.pest_compose_file} not found, skipping local containers")
            return False

        # Start containers with docker compose
        cmd = [
            "docker", "compose",
            "-f", str(compose_file),
            "up", "-d",
            "--scale", f"pest-local-agent={num_agents}",
            "--remove-orphans"
        ]

        publish_fn(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        if result.returncode != 0:
            publish_fn(f"Warning: Failed to start local containers: {result.stderr}")
            return False

        publish_fn(f"Started {num_agents} local agent containers")

        # Update Redis tracking - containers started
        agent_key = f"pest:{run_id}:agents"
        redis_client.hset(agent_key, "local_agents_expected", num_agents)

        return True

    except subprocess.TimeoutExpired:
        publish_fn("Warning: Timeout starting local containers")
        return False
    except Exception as e:
        publish_fn(f"Warning: Error starting local containers: {e}")
        return False


def _stop_local_agent_containers(publish_fn) -> None:
    """Stop local PEST++ agent containers."""
    try:
        compose_file = Path("/app").parent / settings.pest_compose_file
        if not compose_file.exists():
            compose_file = Path(settings.pest_compose_file)

        if not compose_file.exists():
            return

        cmd = [
            "docker", "compose",
            "-f", str(compose_file),
            "down", "--remove-orphans"
        ]

        publish_fn("Stopping local agent containers...")

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        publish_fn("Local agent containers stopped")

    except Exception as e:
        publish_fn(f"Warning: Error stopping local containers: {e}")


def _upload_workspace_for_agents(
    workspace_dir: Path,
    project_id: str,
    run_id: str,
    storage,
    publish_fn,
) -> str:
    """
    Upload PEST workspace to MinIO for remote agents to download.

    Returns the MinIO path prefix where files were uploaded.
    """
    workspace_prefix = f"projects/{project_id}/pest/{run_id}/workspace"

    publish_fn(f"Uploading workspace for remote agents to: {workspace_prefix}")

    file_count = 0
    for f in workspace_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(workspace_dir)
            obj_name = f"{workspace_prefix}/{rel_path}"
            try:
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    obj_name,
                    f.read_bytes(),
                )
                file_count += 1
            except Exception as e:
                publish_fn(f"  Warning: Failed to upload {rel_path}: {e}")

    publish_fn(f"Uploaded {file_count} files to MinIO")
    return workspace_prefix


def _run_pest_network_mode(
    pst_path: Path,
    workspace_dir: Path,
    num_workers: int,
    pestpp_exe: str,
    publish_fn,
    redis_client,
    run_id: str,
    project_id: str,
    storage,
    local_agents: int = 0,
    remote_agents: int = 0,
) -> int:
    """
    Run PEST++ manager in network/hybrid mode with local and/or remote agents.

    This mode supports:
    - Local container agents (via Docker Compose on main server)
    - Remote agents (from other machines on the network)
    - Hybrid mode (both local and remote agents)

    Args:
        pst_path: Path to the PST control file
        workspace_dir: The PEST workspace directory
        num_workers: Total expected workers (local + remote)
        pestpp_exe: Path to the PEST++ executable
        publish_fn: Function to publish messages to Redis
        redis_client: Redis client for cancellation and agent tracking
        run_id: Run ID for tracking
        project_id: Project ID for MinIO path
        storage: Storage service for MinIO uploads
        local_agents: Number of local Docker container agents to start
        remote_agents: Number of remote agents expected from network

    Returns:
        Return code from PEST++ (0 = success)
    """
    # Determine total agents
    total_agents = local_agents + remote_agents
    if total_agents == 0:
        total_agents = num_workers
        # Split between local and remote if not specified
        if settings.pest_local_containers and local_agents == 0:
            local_agents = min(num_workers, settings.pest_local_agents)
            remote_agents = max(0, num_workers - local_agents)
        else:
            remote_agents = num_workers

    publish_fn("=" * 60)
    if local_agents > 0 and remote_agents > 0:
        publish_fn("HYBRID MODE: Local containers + Remote agents")
    elif local_agents > 0:
        publish_fn("CONTAINER MODE: Local Docker agent containers")
    else:
        publish_fn("NETWORK MODE: Waiting for remote agents")
    publish_fn("=" * 60)
    publish_fn(f"  Local container agents: {local_agents}")
    publish_fn(f"  Remote network agents:  {remote_agents}")
    publish_fn(f"  Total expected:         {total_agents}")
    publish_fn("=" * 60)

    # Copy workspace to shared volume for local containers
    if local_agents > 0:
        _copy_workspace_to_shared_volume(workspace_dir, publish_fn)

    # Upload workspace for remote agents
    workspace_prefix = None
    if remote_agents > 0:
        workspace_prefix = _upload_workspace_for_agents(
            workspace_dir, project_id, run_id, storage, publish_fn
        )

        # Store workspace path for agents to find
        storage.upload_bytes(
            settings.minio_bucket_models,
            f"projects/{project_id}/pest/{run_id}/storage_path.txt",
            workspace_prefix.encode("utf-8"),
        )

    # Set up agent tracking in Redis
    agent_key = f"pest:{run_id}:agents"
    redis_client.delete(agent_key)  # Clear any old data
    redis_client.hset(agent_key, "expected", total_agents)
    redis_client.hset(agent_key, "local_agents_expected", local_agents)
    redis_client.hset(agent_key, "local_agents_connected", 0)
    redis_client.hset(agent_key, "remote_agents_expected", remote_agents)
    redis_client.hset(agent_key, "remote_agents_connected", 0)
    redis_client.hset(agent_key, "connected", 0)
    redis_client.hset(agent_key, "status", "waiting")
    redis_client.expire(agent_key, 86400)  # Expire after 24 hours

    # Store run info for agents to find
    run_info = {
        "run_id": run_id,
        "project_id": project_id,
        "workspace_prefix": workspace_prefix,
        "pst_file": pst_path.name,
        "manager_port": settings.pest_manager_port,
        "local_agents": local_agents,
        "remote_agents": remote_agents,
    }
    redis_client.set(
        f"pest:current_run",
        json.dumps(run_info),
        ex=86400,
    )

    pst_name = pst_path.name

    publish_fn(f"Manager will listen on port {settings.pest_manager_port}")
    if remote_agents > 0:
        publish_fn("-" * 60)
        publish_fn("Remote agents can connect using:")
        publish_fn(f"  MANAGER_HOST=<this-server-ip>")
        publish_fn(f"  MANAGER_PORT={settings.pest_manager_port}")
        publish_fn(f"  PROJECT_ID={project_id}")
        publish_fn(f"  RUN_ID={run_id}")
    publish_fn("-" * 60)

    # Start local container agents
    local_started = False
    if local_agents > 0:
        local_started = _start_local_agent_containers(
            local_agents, publish_fn, redis_client, run_id
        )

    # Thread to monitor for cancellation
    cancel_flag = threading.Event()
    process = None

    def check_cancel():
        while not cancel_flag.is_set():
            if redis_client.exists(f"cancel:{run_id}"):
                cancel_flag.set()
                if process:
                    process.terminate()
                break
            cancel_flag.wait(2.0)

    cancel_thread = threading.Thread(target=check_cancel, daemon=True)
    cancel_thread.start()

    return_code = -1

    try:
        # Run PEST++ as manager (listens for agents)
        # The /h :port syntax makes it listen as manager
        cmd = [pestpp_exe, pst_name, f"/h :{settings.pest_manager_port}"]
        publish_fn(f"Starting manager: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            cwd=str(workspace_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        redis_client.hset(agent_key, "status", "running")

        # Monitor output and track agents
        connected_agents = set()
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            publish_fn(line)

            # Parse agent connection messages
            # PEST++ outputs lines like "agent connected from 192.168.1.x"
            if "agent" in line.lower() and "connect" in line.lower():
                # Extract agent info if possible
                connected_agents.add(line)
                redis_client.hset(agent_key, "connected", len(connected_agents))

            if cancel_flag.is_set():
                break

        return_code = process.wait()

        redis_client.hset(agent_key, "status", "completed")

        publish_fn("-" * 60)
        if return_code == 0:
            publish_fn("PEST++ network mode completed successfully")
        else:
            publish_fn(f"PEST++ failed with return code {return_code}")

    except Exception as e:
        publish_fn(f"Error during network mode execution: {e}")
        redis_client.hset(agent_key, "status", "failed")
        return_code = 1
    finally:
        cancel_flag.set()
        cancel_thread.join(timeout=1.0)

        # Close subprocess pipe
        if process is not None and process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass

        # Stop local container agents
        if local_started:
            _stop_local_agent_containers(publish_fn)

        if cancel_flag.is_set() and return_code != 0:
            return_code = -1  # Indicate cancellation

    return return_code


@celery_app.task(bind=True, name="app.tasks.calibrate.run_pest_glm")
def run_pest_glm(self, run_id: str, project_id: str, config: dict) -> dict:
    """
    Execute PEST++ GLM calibration.

    This task:
    1. Downloads model files from MinIO to a working directory
    2. Loads observations from MinIO
    3. Builds PEST workspace (PST, TPL, INS, forward_run.py)
    4. Executes pestpp-glm as a subprocess
    5. Streams stdout to Redis pub/sub for live console
    6. Parses results (phi history, parameters, residuals)
    7. Uploads results back to MinIO

    Args:
        run_id: UUID of the run record
        project_id: UUID of the project
        config: PEST++ configuration dict

    Returns:
        Dict with calibration results and status
    """
    redis_client = get_sync_client()
    channel = f"pest:{run_id}:output"
    history_key = f"pest:{run_id}:history"
    storage = get_storage_service()

    def publish(message: str):
        """Publish message to Redis channel and append to history list."""
        try:
            redis_client.publish(channel, message)
            redis_client.rpush(history_key, message)
            redis_client.ltrim(history_key, -20000, -1)
            redis_client.expire(history_key, 86400)
        except Exception as e:
            logger.warning(f"Redis publish error (calibration continues): {e}")

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

        if not project.storage_path:
            run.status = RunStatus.FAILED
            run.error_message = "No model files found"
            db.commit()
            return {"error": "No model files found"}

        model_type = (
            project.model_type.value if project.model_type else "mf6"
        )

        # Update run status
        run.status = RunStatus.RUNNING
        run.started_at = datetime.utcnow()
        run.celery_task_id = self.request.id
        db.commit()

        publish(f"Starting PEST++ GLM calibration (model type: {model_type})...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = temp_path / "model"
            model_dir.mkdir()
            workspace_dir = temp_path / "pest_workspace"
            workspace_dir.mkdir()

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
                    rel_path = obj_name[len(project.storage_path) :].lstrip(
                        "/"
                    )
                    if not rel_path:
                        continue
                    local_path = model_dir / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    file_data = storage.download_file(
                        settings.minio_bucket_models, obj_name
                    )
                    local_path.write_bytes(file_data)
                    file_count += 1

                publish(f"Downloaded {file_count} model files")
            except Exception as e:
                _fail_run(db, run, f"Failed to download model files: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Load observations
            try:
                publish("Loading observations...")
                obs_obj = f"projects/{project_id}/observations/observations.json"
                if not storage.object_exists(
                    settings.minio_bucket_models, obs_obj
                ):
                    _fail_run(
                        db,
                        run,
                        "No observations uploaded. Upload observations first.",
                    )
                    publish(f"ERROR: {run.error_message}")
                    publish("__STATUS__:failed")
                    return {"error": run.error_message}

                obs_data = json.loads(
                    storage.download_file(
                        settings.minio_bucket_models, obs_obj
                    )
                )
                publish(
                    f"Loaded observations: {obs_data.get('n_observations', 0)} measurements"
                )
            except Exception as e:
                _fail_run(db, run, f"Failed to load observations: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Detect NAM file and executable
            from app.tasks.simulate import (
                detect_modflow_executable,
                get_model_nam_file,
            )

            executable = detect_modflow_executable(model_type)
            if not executable:
                _fail_run(
                    db,
                    run,
                    f"MODFLOW executable not found for {model_type}",
                )
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            nam_file = get_model_nam_file(model_dir, model_type)
            if not nam_file:
                _fail_run(db, run, "Could not find model name file")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Build PEST workspace
            try:
                publish("Building PEST++ workspace...")
                from app.services.pest_setup import (
                    build_observations_for_pest,
                    build_pest_workspace,
                )

                # Convert observations to PEST format
                observations = build_observations_for_pest(obs_data)
                publish(f"  {len(observations)} observation targets")

                # Build parameter list from config
                parameters = []
                for p in config.get("parameters", []):
                    parameters.append(
                        {
                            "property": p["property"],
                            "layer": p.get("layer"),
                            "package_type": p.get("package_type", "array"),
                            "approach": p.get("approach", "multiplier"),
                            "initial_value": p.get("initial_value", 1.0),
                            "lower_bound": p.get("lower_bound", 0.01),
                            "upper_bound": p.get("upper_bound", 100.0),
                            "transform": p.get("transform", "log"),
                            "group": p.get("group", p["property"]),
                        }
                    )
                publish(f"  {len(parameters)} adjustable parameters")

                pest_settings = config.get("settings", {})

                # Apply observation weights from config
                obs_weights = config.get("observation_weights", {})
                for obs in observations:
                    well_name = obs.get("well_name", "")
                    if well_name in obs_weights:
                        obs["weight"] = obs_weights[well_name]

                pst_path = build_pest_workspace(
                    workspace_dir=workspace_dir,
                    model_dir=model_dir,
                    model_type=model_type,
                    nam_file=nam_file,
                    executable=executable,
                    parameters=parameters,
                    observations=observations,
                    pest_settings=pest_settings,
                )

                publish(f"PEST workspace built: {pst_path.name}")
            except Exception as e:
                _fail_run(db, run, f"Failed to build PEST workspace: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Find PEST++ executable
            pestpp_exe = shutil.which(settings.pestpp_exe_path)
            if not pestpp_exe:
                pestpp_exe = shutil.which("pestpp-glm")
            if not pestpp_exe:
                _fail_run(
                    db,
                    run,
                    "pestpp-glm executable not found. "
                    f"Checked: {settings.pestpp_exe_path}, pestpp-glm",
                )
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Run PEST++ (network/hybrid, parallel, or sequential mode)
            try:
                num_workers = pest_settings.get("num_workers", 4)
                num_workers = max(1, min(num_workers, settings.pest_max_num_workers))
                network_mode = pest_settings.get("network_mode", False) or settings.pest_network_mode
                local_containers = pest_settings.get("local_containers", False) or settings.pest_local_containers
                local_agents = pest_settings.get("local_agents", 0)
                remote_agents = pest_settings.get("remote_agents", 0)

                if network_mode or local_containers:
                    mode_desc = "HYBRID" if (local_agents > 0 and remote_agents > 0) else (
                        "CONTAINER" if local_agents > 0 else "NETWORK"
                    )
                    publish(f"Running PEST++ GLM in {mode_desc} MODE...")
                    return_code = _run_pest_network_mode(
                        pst_path=pst_path,
                        workspace_dir=workspace_dir,
                        num_workers=num_workers,
                        pestpp_exe=pestpp_exe,
                        publish_fn=publish,
                        redis_client=redis_client,
                        run_id=run_id,
                        project_id=project_id,
                        storage=storage,
                        local_agents=local_agents,
                        remote_agents=remote_agents,
                    )
                elif num_workers > 1:
                    publish(f"Running PEST++ GLM with {num_workers} local parallel workers...")
                    return_code = _run_pest_parallel(
                        pst_path=pst_path,
                        workspace_dir=workspace_dir,
                        num_workers=num_workers,
                        pestpp_exe=pestpp_exe,
                        publish_fn=publish,
                        redis_client=redis_client,
                        run_id=run_id,
                    )
                else:
                    publish("Running PEST++ GLM sequentially (1 worker)...")
                    return_code = _run_pest_sequential(
                        pst_path=pst_path,
                        workspace_dir=workspace_dir,
                        pestpp_exe=pestpp_exe,
                        publish_fn=publish,
                        redis_client=redis_client,
                        run_id=run_id,
                    )

                publish("-" * 60)

                if return_code == -1:
                    # Cancelled
                    run.status = RunStatus.CANCELLED
                    run.error_message = "Cancelled by user"
                    run.completed_at = datetime.utcnow()
                    db.commit()
                    publish("PEST++ calibration cancelled by user")
                    publish("__STATUS__:cancelled")
                    redis_client.delete(f"cancel:{run_id}")
                    return {"status": "cancelled"}
                elif return_code == 0:
                    run.status = RunStatus.COMPLETED
                    run.exit_code = return_code
                    publish("PEST++ GLM calibration completed!")
                else:
                    run.status = RunStatus.FAILED
                    run.exit_code = return_code
                    run.error_message = (
                        f"PEST++ failed with return code {return_code}"
                    )
                    publish(
                        f"PEST++ failed with return code {return_code}"
                    )

                run.completed_at = datetime.utcnow()

            except Exception as e:
                _fail_run(db, run, f"Error running PEST++: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Parse results and upload
            if run.status == RunStatus.COMPLETED:
                try:
                    from app.services.pest_setup import parse_pest_results

                    publish("Parsing calibration results...")
                    results = parse_pest_results(workspace_dir)

                    # Upload results to MinIO
                    output_prefix = (
                        f"{project.storage_path}/pest/{run_id}"
                    )
                    run.results_path = output_prefix

                    # Upload results JSON
                    results_json = json.dumps(results, default=_json_default)
                    storage.upload_bytes(
                        settings.minio_bucket_models,
                        f"{output_prefix}/pest_results.json",
                        results_json.encode("utf-8"),
                        content_type="application/json",
                    )

                    # Upload key PEST++ output files
                    pest_output_exts = {
                        ".rec",
                        ".par",
                        ".rei",
                        ".jco",
                        ".jcb",
                        ".rst",
                    }
                    uploaded = 0
                    for f in workspace_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in pest_output_exts:
                            obj_name = f"{output_prefix}/{f.name}"
                            try:
                                storage.upload_bytes(
                                    settings.minio_bucket_models,
                                    obj_name,
                                    f.read_bytes(),
                                )
                                uploaded += 1
                            except Exception:
                                pass

                    publish(f"Uploaded {uploaded} result files")

                    # Update convergence_info with summary
                    run.convergence_info = {
                        "converged": results.get("converged", False),
                        "n_iterations": len(
                            results.get("phi_history", [])
                        ),
                        "final_phi": (
                            results["phi_history"][-1]["phi"]
                            if results.get("phi_history")
                            else None
                        ),
                        "n_parameters": len(
                            results.get("parameters", {})
                        ),
                        "n_observations": len(
                            results.get("residuals", [])
                        ),
                    }

                except Exception as e:
                    publish(f"Warning: Error parsing results: {e}")

            db.commit()

            final_status = (
                "completed"
                if run.status == RunStatus.COMPLETED
                else "failed"
            )
            publish(f"__STATUS__:{final_status}")

            return {
                "run_id": run_id,
                "status": final_status,
                "return_code": run.exit_code,
            }


@celery_app.task(bind=True, name="app.tasks.calibrate.run_pest_ies")
def run_pest_ies(self, run_id: str, project_id: str, config: dict) -> dict:
    """
    Execute PEST++ IES (Iterative Ensemble Smoother) uncertainty analysis.

    Similar to GLM but uses pestpp-ies executable and parses
    ensemble output files (phi.actual.csv, par.csv, obs.csv).
    """
    redis_client = get_sync_client()
    channel = f"pest:{run_id}:output"
    history_key = f"pest:{run_id}:history"
    storage = get_storage_service()

    def publish(message: str):
        try:
            redis_client.publish(channel, message)
            redis_client.rpush(history_key, message)
            redis_client.ltrim(history_key, -20000, -1)
            redis_client.expire(history_key, 86400)
        except Exception as e:
            logger.warning(f"Redis publish error (calibration continues): {e}")

    with SessionLocal() as db:
        project = db.execute(
            select(Project).where(Project.id == UUID(project_id))
        ).scalar_one_or_none()

        run = db.execute(
            select(Run).where(Run.id == UUID(run_id))
        ).scalar_one_or_none()

        if not project or not run:
            return {"error": "Project or run not found"}

        if not project.storage_path:
            run.status = RunStatus.FAILED
            run.error_message = "No model files found"
            db.commit()
            return {"error": "No model files found"}

        model_type = (
            project.model_type.value if project.model_type else "mf6"
        )

        run.status = RunStatus.RUNNING
        run.started_at = datetime.utcnow()
        run.celery_task_id = self.request.id
        db.commit()

        publish(f"Starting PEST++ IES ensemble analysis (model type: {model_type})...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = temp_path / "model"
            model_dir.mkdir()
            workspace_dir = temp_path / "pest_workspace"
            workspace_dir.mkdir()

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
                    rel_path = obj_name[len(project.storage_path) :].lstrip(
                        "/"
                    )
                    if not rel_path:
                        continue
                    local_path = model_dir / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    file_data = storage.download_file(
                        settings.minio_bucket_models, obj_name
                    )
                    local_path.write_bytes(file_data)
                    file_count += 1

                publish(f"Downloaded {file_count} model files")
            except Exception as e:
                _fail_run(db, run, f"Failed to download model files: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Load observations
            try:
                publish("Loading observations...")
                obs_obj = f"projects/{project_id}/observations/observations.json"
                if not storage.object_exists(
                    settings.minio_bucket_models, obs_obj
                ):
                    _fail_run(
                        db,
                        run,
                        "No observations uploaded. Upload observations first.",
                    )
                    publish(f"ERROR: {run.error_message}")
                    publish("__STATUS__:failed")
                    return {"error": run.error_message}

                obs_data = json.loads(
                    storage.download_file(
                        settings.minio_bucket_models, obs_obj
                    )
                )
                publish(
                    f"Loaded observations: {obs_data.get('n_observations', 0)} measurements"
                )
            except Exception as e:
                _fail_run(db, run, f"Failed to load observations: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Detect NAM file and executable
            from app.tasks.simulate import (
                detect_modflow_executable,
                get_model_nam_file,
            )

            executable = detect_modflow_executable(model_type)
            if not executable:
                _fail_run(
                    db,
                    run,
                    f"MODFLOW executable not found for {model_type}",
                )
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            nam_file = get_model_nam_file(model_dir, model_type)
            if not nam_file:
                _fail_run(db, run, "Could not find model name file")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Build PEST workspace with IES settings
            try:
                publish("Building PEST++ IES workspace...")
                from app.services.pest_setup import (
                    build_observations_for_pest,
                    build_pest_workspace,
                )

                observations = build_observations_for_pest(obs_data)
                publish(f"  {len(observations)} observation targets")

                parameters = []
                for p in config.get("parameters", []):
                    parameters.append(
                        {
                            "property": p["property"],
                            "layer": p.get("layer"),
                            "package_type": p.get("package_type", "array"),
                            "approach": p.get("approach", "multiplier"),
                            "initial_value": p.get("initial_value", 1.0),
                            "lower_bound": p.get("lower_bound", 0.01),
                            "upper_bound": p.get("upper_bound", 100.0),
                            "transform": p.get("transform", "log"),
                            "group": p.get("group", p["property"]),
                        }
                    )
                publish(f"  {len(parameters)} adjustable parameters")

                pest_settings = config.get("settings", {})
                # Mark as IES method for PST file generation
                pest_settings["method"] = "ies"

                obs_weights = config.get("observation_weights", {})
                for obs in observations:
                    well_name = obs.get("well_name", "")
                    if well_name in obs_weights:
                        obs["weight"] = obs_weights[well_name]

                ies_num_reals = pest_settings.get("ies_num_reals", 50)
                publish(f"  Ensemble size: {ies_num_reals} realizations")

                pst_path = build_pest_workspace(
                    workspace_dir=workspace_dir,
                    model_dir=model_dir,
                    model_type=model_type,
                    nam_file=nam_file,
                    executable=executable,
                    parameters=parameters,
                    observations=observations,
                    pest_settings=pest_settings,
                )

                publish(f"PEST workspace built: {pst_path.name}")
            except Exception as e:
                _fail_run(db, run, f"Failed to build PEST workspace: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Find PEST++ IES executable
            pestpp_exe = shutil.which(settings.pestpp_ies_exe_path)
            if not pestpp_exe:
                pestpp_exe = shutil.which("pestpp-ies")
            if not pestpp_exe:
                _fail_run(
                    db,
                    run,
                    "pestpp-ies executable not found. "
                    f"Checked: {settings.pestpp_ies_exe_path}, pestpp-ies",
                )
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Run PEST++ IES (network/hybrid, parallel, or sequential mode)
            try:
                num_workers = pest_settings.get("num_workers", 4)
                num_workers = max(1, min(num_workers, settings.pest_max_num_workers))
                network_mode = pest_settings.get("network_mode", False) or settings.pest_network_mode
                local_containers = pest_settings.get("local_containers", False) or settings.pest_local_containers
                local_agents = pest_settings.get("local_agents", 0)
                remote_agents = pest_settings.get("remote_agents", 0)

                if network_mode or local_containers:
                    mode_desc = "HYBRID" if (local_agents > 0 and remote_agents > 0) else (
                        "CONTAINER" if local_agents > 0 else "NETWORK"
                    )
                    publish(f"Running PEST++ IES in {mode_desc} MODE...")
                    return_code = _run_pest_network_mode(
                        pst_path=pst_path,
                        workspace_dir=workspace_dir,
                        num_workers=num_workers,
                        pestpp_exe=pestpp_exe,
                        publish_fn=publish,
                        redis_client=redis_client,
                        run_id=run_id,
                        project_id=project_id,
                        storage=storage,
                        local_agents=local_agents,
                        remote_agents=remote_agents,
                    )
                elif num_workers > 1:
                    publish(f"Running PEST++ IES with {num_workers} local parallel workers...")
                    return_code = _run_pest_parallel(
                        pst_path=pst_path,
                        workspace_dir=workspace_dir,
                        num_workers=num_workers,
                        pestpp_exe=pestpp_exe,
                        publish_fn=publish,
                        redis_client=redis_client,
                        run_id=run_id,
                    )
                else:
                    publish("Running PEST++ IES sequentially (1 worker)...")
                    return_code = _run_pest_sequential(
                        pst_path=pst_path,
                        workspace_dir=workspace_dir,
                        pestpp_exe=pestpp_exe,
                        publish_fn=publish,
                        redis_client=redis_client,
                        run_id=run_id,
                    )

                publish("-" * 60)

                if return_code == -1:
                    # Cancelled
                    run.status = RunStatus.CANCELLED
                    run.error_message = "Cancelled by user"
                    run.completed_at = datetime.utcnow()
                    db.commit()
                    publish("PEST++ IES analysis cancelled by user")
                    publish("__STATUS__:cancelled")
                    redis_client.delete(f"cancel:{run_id}")
                    return {"status": "cancelled"}
                elif return_code == 0:
                    run.status = RunStatus.COMPLETED
                    run.exit_code = return_code
                    publish("PEST++ IES ensemble analysis completed!")
                else:
                    run.status = RunStatus.FAILED
                    run.exit_code = return_code
                    run.error_message = (
                        f"PEST++ IES failed with return code {return_code}"
                    )
                    publish(
                        f"PEST++ IES failed with return code {return_code}"
                    )

                run.completed_at = datetime.utcnow()

            except Exception as e:
                _fail_run(db, run, f"Error running PEST++ IES: {e}")
                publish(f"ERROR: {run.error_message}")
                publish("__STATUS__:failed")
                return {"error": run.error_message}

            # Parse results and upload
            if run.status == RunStatus.COMPLETED:
                try:
                    from app.services.pest_setup import parse_ies_results

                    publish("Parsing IES ensemble results...")
                    results = parse_ies_results(workspace_dir)

                    output_prefix = (
                        f"{project.storage_path}/pest/{run_id}"
                    )
                    run.results_path = output_prefix

                    results_json = json.dumps(results, default=_json_default)
                    storage.upload_bytes(
                        settings.minio_bucket_models,
                        f"{output_prefix}/pest_results.json",
                        results_json.encode("utf-8"),
                        content_type="application/json",
                    )

                    # Upload key PEST++ IES output files
                    pest_output_exts = {
                        ".rec",
                        ".par",
                        ".rei",
                        ".csv",
                        ".jco",
                        ".jcb",
                    }
                    uploaded = 0
                    for f in workspace_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in pest_output_exts:
                            obj_name = f"{output_prefix}/{f.name}"
                            try:
                                storage.upload_bytes(
                                    settings.minio_bucket_models,
                                    obj_name,
                                    f.read_bytes(),
                                )
                                uploaded += 1
                            except Exception:
                                pass

                    publish(f"Uploaded {uploaded} result files")

                    n_reals = results.get("ensemble", {}).get("n_reals", 0)
                    n_failed = results.get("ensemble", {}).get("n_failed", 0)
                    run.convergence_info = {
                        "converged": results.get("converged", False),
                        "n_iterations": len(
                            results.get("phi_history", [])
                        ),
                        "final_phi": (
                            results["phi_history"][-1]["phi"]
                            if results.get("phi_history")
                            else None
                        ),
                        "n_parameters": len(
                            results.get("parameters", {})
                        ),
                        "n_observations": len(
                            results.get("residuals", [])
                        ),
                        "n_realizations": n_reals,
                        "n_failed_realizations": n_failed,
                    }

                except Exception as e:
                    publish(f"Warning: Error parsing results: {e}")

            db.commit()

            final_status = (
                "completed"
                if run.status == RunStatus.COMPLETED
                else "failed"
            )
            publish(f"__STATUS__:{final_status}")

            return {
                "run_id": run_id,
                "status": final_status,
                "return_code": run.exit_code,
            }


def _fail_run(db: Session, run: Run, message: str) -> None:
    """Mark a run as failed with an error message."""
    run.status = RunStatus.FAILED
    run.error_message = message
    run.completed_at = datetime.utcnow()
    db.commit()


def _json_default(obj):
    """JSON serializer for numpy types."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
