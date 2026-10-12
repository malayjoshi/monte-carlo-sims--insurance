import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _read_key_value_file(path: str) -> dict[str, str]:
    values = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


def _read_os_release() -> dict[str, str]:
    values = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip('"')
    except OSError:
        pass
    return values


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment() -> dict:
    cpuinfo = _read_key_value_file("/proc/cpuinfo")
    meminfo = _read_key_value_file("/proc/meminfo")
    core_ids = set()
    current_physical = current_core = None
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines() + [""]
        for line in lines:
            if not line:
                if current_physical is not None and current_core is not None:
                    core_ids.add((current_physical, current_core))
                current_physical = current_core = None
            elif line.startswith("physical id"):
                current_physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                current_core = line.split(":", 1)[1].strip()
    except OSError:
        pass
    blas_output = io.StringIO()
    with contextlib.redirect_stdout(blas_output), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.show_config()
    os_release = _read_os_release()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": cpuinfo.get("model name", platform.processor() or "unknown"),
        "physical_cpu_cores": len(core_ids) or None,
        "logical_cpu_count": os.cpu_count(),
        "total_ram": meminfo.get("MemTotal", "unknown"),
        "linux_distribution": os_release.get("PRETTY_NAME", "unknown"),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "blas_configuration": blas_output.getvalue(),
        "threading_environment": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "git_commit_sha": _git_sha(),
    }


def write_environment(path: Path) -> dict:
    metadata = collect_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
