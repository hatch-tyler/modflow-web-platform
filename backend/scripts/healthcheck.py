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
import signal
import sys


def check_api() -> bool:
    """Check if the FastAPI server is responding.

    Also detects the "zombie worker" state where uvicorn's reloader (PID 1)
    is alive but its worker subprocess has crashed and become defunct.
    In this state, all HTTP requests hang indefinitely because there is no
    worker to serve them, and the reloader won't restart without a file change.
    """
    import urllib.request
    import urllib.error

    # Detect "dead worker" state: uvicorn's reloader is alive but its worker
    # subprocess has crashed (zombie or fully reaped by tini). In this state
    # every HTTP request hangs forever because no worker is serving.
    #
    # How this happens:
    #  1. --reload watcher detects file change (host volume mount)
    #  2. Reloader kills old worker, forks new one
    #  3. New worker crashes on import (partial file write, syntax error, etc.)
    #  4. Reloader has no process.is_alive() check — it blocks waiting for
    #     the *next* file change, never noticing the worker is dead.
    #
    # Fix: detect the missing worker and SIGTERM PID 1 to restart the container.
    try:
        import subprocess
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            # Count live python processes (exclude: ps itself, healthcheck, zombies)
            live_python_procs = 0
            has_zombie = False
            for line in lines[1:]:
                cols = line.split()
                if len(cols) < 8:
                    continue
                stat = cols[7] if len(cols) > 7 else ""
                cmd = " ".join(cols[10:]) if len(cols) > 10 else ""
                if "python" not in cmd and "uvicorn" not in cmd:
                    continue
                if "healthcheck" in cmd or "ps aux" in cmd:
                    continue
                if stat.startswith("Z") or "<defunct>" in line:
                    has_zombie = True
                    continue
                live_python_procs += 1

            # With --reload, we expect at least 2 python processes:
            # the reloader supervisor and the worker. If only 1 (the reloader)
            # or if there's a zombie and only the reloader, the worker is dead.
            if live_python_procs <= 1 and (has_zombie or live_python_procs == 1):
                # Only kill if the container has been up long enough to have started
                # (avoid false positives during startup)
                uptime_file = "/proc/1/stat"
                try:
                    with open(uptime_file, "r") as f:
                        # Field 22 is starttime in jiffies
                        stat_fields = f.read().split()
                        start_jiffies = int(stat_fields[21])
                        hz = os.sysconf("SC_CLK_TCK")
                        with open("/proc/uptime", "r") as uf:
                            system_uptime = float(uf.read().split()[0])
                        proc_uptime = system_uptime - (start_jiffies / hz)
                        if proc_uptime < 30:
                            pass  # Still starting up, don't kill
                        else:
                            print(
                                f"Uvicorn worker is dead (live_procs={live_python_procs}, "
                                f"zombie={has_zombie}, uptime={proc_uptime:.0f}s). "
                                f"Sending SIGTERM to trigger container restart.",
                                file=sys.stderr,
                            )
                            os.kill(1, signal.SIGTERM)
                            return False
                except Exception:
                    # Can't read proc, fall through to HTTP check
                    pass
    except Exception:
        pass  # If process check fails, fall through to HTTP check

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
