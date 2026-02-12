#!/usr/bin/env python3
"""
Health check script for Docker containers.

Supports both API (HTTP) and Worker (Celery) health checks.
Usage:
    python scripts/healthcheck.py api     # Check API server
    python scripts/healthcheck.py worker  # Check Celery worker
    python scripts/healthcheck.py         # Auto-detect based on process
"""

import os
import sys


def check_api() -> bool:
    """Check if the FastAPI server is responding."""
    import urllib.request
    import urllib.error

    try:
        url = "http://localhost:8000/api/v1/health/live"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception as e:
        print(f"API health check failed: {e}", file=sys.stderr)
        return False


def check_worker() -> bool:
    """Check if the Celery worker is running and can process tasks."""
    try:
        from celery_app import celery_app

        # Ping the worker - this checks if it's connected to the broker
        # and can respond to control commands
        inspect = celery_app.control.inspect()

        # Get active queues - if we get any response, worker is alive
        ping_response = inspect.ping()

        if ping_response:
            print(f"Worker responding: {list(ping_response.keys())}")
            return True
        else:
            # No workers responded - check if broker is at least reachable
            try:
                with celery_app.connection_or_acquire() as conn:
                    conn.ensure_connection(max_retries=1)
                print("Broker reachable but no workers responding yet")
                # Return True if broker is reachable - worker might still be starting
                return True
            except Exception as e:
                print(f"Broker not reachable: {e}", file=sys.stderr)
                return False

    except Exception as e:
        print(f"Worker health check failed: {e}", file=sys.stderr)
        return False


def check_worker_simple() -> bool:
    """Simple worker check - just verify Redis broker is reachable."""
    try:
        import redis
        from app.config import get_settings

        settings = get_settings()
        r = redis.from_url(settings.redis_url, socket_timeout=5)
        r.ping()
        r.close()
        return True
    except Exception as e:
        print(f"Worker health check (simple) failed: {e}", file=sys.stderr)
        return False


def detect_mode() -> str:
    """Detect whether we're running as API or worker based on environment."""
    # Check for common indicators
    # In docker-compose, we could set an explicit env var
    mode = os.environ.get("HEALTH_CHECK_MODE", "").lower()
    if mode in ("api", "worker"):
        return mode

    # Try to detect from running processes
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn"],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            return "api"
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "celery"],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            return "worker"
    except Exception:
        pass

    # Default to API
    return "api"


def main():
    # Get mode from argument or auto-detect
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = detect_mode()

    print(f"Health check mode: {mode}")

    if mode == "api":
        success = check_api()
    elif mode == "worker":
        # Use simple check (Redis connectivity) for faster response
        success = check_worker_simple()
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
