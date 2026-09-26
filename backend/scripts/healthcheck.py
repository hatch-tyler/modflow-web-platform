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

# The API health check can SIGTERM PID 1 to force a container restart when the
# uvicorn worker is gone. That is a blunt instrument, so it must not fire on a
# single sample. State lives in /tmp (container-local, cleared on restart), so
# strikes naturally reset when the container comes back.
_STRIKE_FILE = "/tmp/.healthcheck_api_strikes"
_STRIKES_BEFORE_KILL = 3


def _bump_strikes() -> int:
    """Increment and return the consecutive-failure count."""
    try:
        with open(_STRIKE_FILE, "r") as f:
            count = int(f.read().strip() or 0)
    except Exception:
        count = 0
    count += 1
    try:
        with open(_STRIKE_FILE, "w") as f:
            f.write(str(count))
    except Exception:
        # If we cannot persist state, fail safe: never reach the kill
        # threshold on the basis of a count we could not record.
        return 1
    return count


def _reset_strikes() -> None:
    try:
        os.remove(_STRIKE_FILE)
    except Exception:
        pass


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
                    # Require the condition to PERSIST before killing PID 1.
                    # A single bad sample is not proof of a dead worker: the
                    # reloader briefly runs one process while respawning. Only
                    # a sustained shortage means the worker is really gone.
                    strikes = _bump_strikes()
                    if strikes < _STRIKES_BEFORE_KILL:
                        print(
                            f"Uvicorn worker may be dead "
                            f"(live_workers={live_workers}, zombie={has_zombie}, "
                            f"strike {strikes}/{_STRIKES_BEFORE_KILL}). "
                            f"Not restarting yet.",
                            file=sys.stderr,
                        )
                        return False
                    print(
                        f"Uvicorn worker is dead (live_workers={live_workers}, "
                        f"zombie={has_zombie}, uptime={proc_uptime:.0f}s, "
                        f"{strikes} consecutive strikes). "
                        f"Sending SIGTERM to trigger container restart.",
                        file=sys.stderr,
                    )
                    _reset_strikes()
                    os.kill(1, signal.SIGTERM)
                    return False
            except Exception:
                # Can't read proc uptime, fall through to HTTP check
                pass
        else:
            # Healthy process count — clear any accumulated strikes.
            _reset_strikes()
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
    """Check that the Celery worker in THIS container is accepting work.

    This is the real liveness probe. It asks the worker itself to respond to
    a control-plane ping, which the MainProcess consumer answers even while a
    prefork child is busy running a multi-hour MODFLOW simulation.

    It deliberately does NOT fall back to "broker is reachable" as a success
    condition. That fallback made the check report healthy whenever Redis was
    up — including when the Celery process was dead or wedged — which is the
    exact state the health check exists to catch.
    """
    import socket

    try:
        from celery_app import celery_app

        host = socket.gethostname()
        response = celery_app.control.inspect(timeout=5.0).ping() or {}

        # Require a response from THIS container's worker. Accepting any
        # responder would let a sibling worker (or a PEST agent sharing the
        # broker) mask a local failure — the same blind spot as the old
        # Redis-only check, one layer up. Match on hostname rather than an
        # exact "celery@<host>" so an explicit -n prefix still resolves.
        mine = [n for n in response if n.endswith(f"@{host}") or host in n]

        if mine:
            print(f"Worker responding: {mine}")
            return True

        if response:
            print(
                f"Other workers responded {list(response)} but not this "
                f"container ({host}) — local worker is not accepting work.",
                file=sys.stderr,
            )
        else:
            print(
                f"No Celery worker responded to ping (expected host {host}). "
                f"Broker may be up, but this worker is not accepting work.",
                file=sys.stderr,
            )
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
        # Ping the Celery worker itself. check_worker_simple() (Redis-only) is
        # kept below as a diagnostic helper but is NOT the health check: it
        # cannot distinguish "worker alive" from "worker dead, Redis fine".
        success = check_worker()
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
