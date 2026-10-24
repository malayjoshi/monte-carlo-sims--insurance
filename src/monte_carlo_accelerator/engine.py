"""Backend selection for Monte Carlo projections."""

from typing import Literal

from .cpu_engine import project_cpu
from .gpu_engine import GPUUnavailableError, project_gpu
from .gpu_worker import WarmGPUWorker
from .models import Policies, ProjectionConfig, ProjectionResult

Backend = Literal["auto", "cpu", "gpu"]

# Conservative policy derived from the 2026-10-24 g4dn.xlarge/Tesla T4 run.
# This is configurable because the crossover depends on the CPU, GPU, and process
# warmth; it is not a universal hardware constant.
DEFAULT_GPU_THRESHOLD = 10_000_000


def project(
    policies: Policies,
    scenarios: int,
    config: ProjectionConfig = ProjectionConfig(),
    *,
    backend: Backend = "auto",
    gpu_threshold: int = DEFAULT_GPU_THRESHOLD,
    gpu_worker: WarmGPUWorker | None = None,
) -> ProjectionResult:
    """Project with an explicit backend or choose one from workload size.

    Automatic dispatch avoids importing/initializing CUDA below ``gpu_threshold``.
    At or above the threshold it attempts the GPU and falls back to the CPU only
    when CuPy/CUDA is unavailable. GPU execution errors are intentionally exposed.
    """
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    if gpu_threshold <= 0:
        raise ValueError("gpu_threshold must be positive")

    if backend == "cpu" or (backend == "auto" and len(policies) * scenarios < gpu_threshold):
        return project_cpu(policies, scenarios, config)
    gpu_project = gpu_worker.project if gpu_worker is not None else project_gpu
    if backend == "gpu":
        return gpu_project(policies, scenarios, config)

    try:
        return gpu_project(policies, scenarios, config)
    except GPUUnavailableError:
        return project_cpu(policies, scenarios, config)
