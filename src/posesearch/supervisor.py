from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    # Avoid hidden thread multiplication on small NAS CPUs.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def main() -> None:
    env = _child_env()
    api = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "posesearch.api:app",
        "--host", "0.0.0.0", "--port", os.getenv("PORT", "8080"),
        "--workers", "1",
    ], env=env)
    worker = subprocess.Popen([sys.executable, "-m", "posesearch.worker"], env=env)
    children = [api, worker]
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    exit_code = 0
    try:
        while True:
            for child in children:
                rc = child.poll()
                if rc is not None:
                    exit_code = rc
                    stop(signal.SIGTERM, None)
                    raise SystemExit(exit_code)
            time.sleep(0.5)
    finally:
        deadline = time.time() + 10
        for child in children:
            remaining = max(0.1, deadline - time.time())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
