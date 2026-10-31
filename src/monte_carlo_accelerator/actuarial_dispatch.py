"""Backend selection for Stage 3 term-life projections."""

from typing import Literal

from .actuarial_engine import project_term_life
from .actuarial_gpu_engine import project_term_life_gpu
from .gpu_engine import GPUUnavailableError
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies

Stage3Backend = Literal["auto", "cpu", "gpu"]

# Conservative crossover derived from the 2026-10-24 g4dn.xlarge/Tesla T4
# benchmark. Keep it configurable because hardware and process warmth matter.
DEFAULT_STAGE3_GPU_THRESHOLD = 10_000_000


def project_term_life_auto(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
    *,
    backend: Stage3Backend = "auto",
    gpu_threshold: int = DEFAULT_STAGE3_GPU_THRESHOLD,
) -> ActuarialProjectionResult:
    """Project Stage 3 cash flows with an explicit or automatic backend.

    Automatic dispatch avoids importing or initializing CUDA below the
    threshold. At or above it, GPU unavailability safely falls back to CPU.
    Other GPU execution errors remain visible instead of being hidden.
    """
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    if gpu_threshold <= 0:
        raise ValueError("gpu_threshold must be positive")

    workload = len(policies) * scenarios
    if backend == "cpu" or (backend == "auto" and workload < gpu_threshold):
        return project_term_life(policies, scenarios, assumptions)
    if backend == "gpu":
        return project_term_life_gpu(policies, scenarios, assumptions)

    try:
        return project_term_life_gpu(policies, scenarios, assumptions)
    except GPUUnavailableError:
        return project_term_life(policies, scenarios, assumptions)
