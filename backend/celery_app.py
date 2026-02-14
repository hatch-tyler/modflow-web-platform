"""Celery application configuration."""

from celery import Celery
from celery.signals import worker_ready, worker_shutdown, worker_shutting_down

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "modflow_worker",
    broker=settings.get_celery_broker_url(),
    backend=settings.get_celery_result_backend(),
    include=[
        "app.tasks.simulate",
        "app.tasks.calibrate",
        "app.tasks.postprocess",
        "app.tasks.live_results",
        "app.tasks.zonebudget",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution settings
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    worker_prefetch_multiplier=1,  # One task at a time for long-running jobs
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    # Default queue (all tasks go to default for now)
    task_default_queue="default",
    # Recycle worker process every 100 tasks to clean up accumulated leaks
    worker_max_tasks_per_child=100,
)


@worker_ready.connect
def cleanup_orphaned_runs(sender=None, **kwargs):
    """
    On worker startup, find any runs stuck in RUNNING status and mark them as failed.

    This handles the case where a previous worker was OOM-killed (SIGKILL) mid-simulation,
    leaving runs permanently stuck as 'running' with no process to complete them.
    """
    import logging
    from datetime import datetime, timezone

    from sqlalchemy import select

    from app.models.project import Run, RunStatus
    from app.models.base import SessionLocal
    from app.services.redis_manager import get_sync_client

    logger = logging.getLogger(__name__)

    try:
        redis_client = get_sync_client()
        with SessionLocal() as db:
            orphaned = db.execute(
                select(Run).where(Run.status == RunStatus.RUNNING)
            ).scalars().all()

            for run in orphaned:
                # Check if there's an active execution lock — if so, another worker
                # may genuinely be running it (shouldn't happen with concurrency=1, but safe)
                lock_key = f"simulation_lock:{run.id}"
                if redis_client.exists(lock_key):
                    # Lock exists but worker just started — the lock holder is dead
                    redis_client.delete(lock_key)

                run.status = RunStatus.FAILED
                run.error_message = "Worker was terminated during simulation (likely OOM killed)"
                run.completed_at = datetime.now(timezone.utc)
                logger.warning(f"Marked orphaned run {run.id} as failed")

                # Publish status to Redis so any SSE listeners get notified
                channel = f"simulation:{run.id}:output"
                history_key = f"simulation:{run.id}:history"
                msg = "Worker was terminated during simulation (likely OOM killed)"
                redis_client.publish(channel, msg)
                redis_client.rpush(history_key, msg)
                redis_client.publish(channel, "__STATUS__:failed")
                redis_client.rpush(history_key, "__STATUS__:failed")
                redis_client.expire(history_key, 86400)

            db.commit()

            if orphaned:
                logger.warning(f"Cleaned up {len(orphaned)} orphaned running run(s)")
    except Exception as e:
        logger.error(f"Failed to clean up orphaned runs: {e}")


@worker_shutting_down.connect
def graceful_shutdown(sender=None, sig=None, how=None, exitcode=None, **kwargs):
    """Handle graceful worker shutdown.

    When the worker receives SIGTERM (e.g. docker stop, deploy), Celery will:
    1. Stop accepting new tasks
    2. Wait for currently executing tasks to finish (up to CELERY_WORKER_TERM_TIMEOUT)
    3. Then call worker_shutdown

    We log the event so admins can see the shutdown was graceful.
    The actual waiting is handled by Celery's warm shutdown mechanism.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"Worker shutting down (signal={sig}, how={how}). "
        "Waiting for current task to complete..."
    )


@worker_shutdown.connect
def cleanup_on_shutdown(sender=None, **kwargs):
    """Close shared Redis connection pool on worker shutdown."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        from app.services.redis_manager import close_all_sync
        close_all_sync()
        logger.info("Redis connection pool closed on worker shutdown")
    except Exception as e:
        logger.warning(f"Error closing Redis pool on shutdown: {e}")


if __name__ == "__main__":
    celery_app.start()
