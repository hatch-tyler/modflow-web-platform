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

    # Detect "dead worker" state using /proc filesystem inspection.
    # Uvicorn's reloader (PID 1 or child of tini) is alive but its worker
    # subprocess has crashed (zombie or fully reaped). In this state every
    # HTTP request hangs forever because no worker is serving.
    #
    # Previous approach parsed `ps aux` output by column index, which broke
    # when column positions shifted due to username length, timestamps, etc.
    # /proc/[pid]/status provides structured, unambiguous process state.
    try:
        live_workers = 0
        has_zombie = False

        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = entry
            try:
                # Read process state from /proc/[pid]/status
                with open(f"/proc/{pid}/status", "r") as f:
                    status_lines = f.read()

                state = ""
                for line in status_lines.splitlines():
                    if line.startswith("State:"):
                        # Format: "State:\tS (sleeping)" — grab the letter
                        state = line.split("\t", 1)[1][0] if "\t" in line else ""
                        break

                # Read command line from /proc/[pid]/cmdline (null-separated)
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace("\0", " ").strip()

                # Only consider python/uvicorn processes — check the executable
                # (first token), not just args, to avoid counting tini/docker-init
                # whose cmdline includes "uvicorn" as a passthrough argument.
                exe = cmdline.split()[0] if cmdline else ""
                if "python" not in exe and "uvicorn" not in exe:
                    continue
                # Skip our own healthcheck process and the multiprocessing
                # resource_tracker (bookkeeping helper, not an HTTP worker).
                if "healthcheck" in cmdline or "resource_tracker" in cmdline:
                    continue

                if state == "Z":
                    has_zombie = True
                else:
                    live_workers += 1
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                # Process exited between listdir and read — skip
                continue

        # With --reload, we expect at least 2 python processes:
        # the reloader supervisor and the worker. If only 1 (the reloader)
        # or 0, and especially if there's a zombie, the worker is dead.
        if live_workers <= 1:
            # Only kill if the container has been up long enough to have started
            # (avoid false positives during startup)
            try:
                with open("/proc/1/stat", "r") as f:
                    stat_fields = f.read().split()
                    start_jiffies = int(stat_fields[21])
                hz = os.sysconf("SC_CLK_TCK")
                with open("/proc/uptime", "r") as uf:
                    system_uptime = float(uf.read().split()[0])
                proc_uptime = system_uptime - (start_jiffies / hz)
                if proc_uptime >= 30:
                    print(
                        f"Uvicorn worker is dead (live_workers={live_workers}, "
                        f"zombie={has_zombie}, uptime={proc_uptime:.0f}s). "
                        f"Sending SIGTERM to trigger container restart.",
                        file=sys.stderr,
                    )
                    os.kill(1, signal.SIGTERM)
                    return False
            except Exception:
                # Can't read proc uptime, fall through to HTTP check
                pass
    except Exception:
        pass  # If /proc inspection fails, fall through to HTTP check

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
