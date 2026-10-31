"""Monte Carlo accelerator."""

from .actuarial_engine import project_term_life
from .actuarial_dispatch import (
    DEFAULT_STAGE3_GPU_THRESHOLD,
    project_term_life_auto,
)
from .actuarial_gpu_engine import (
    GpuTermLifePolicies,
    prepare_gpu_term_life_policies,
    project_term_life_gpu,
    project_term_life_gpu_prepared,
)
from .engine import DEFAULT_GPU_THRESHOLD, project
from .gpu_worker import WarmGPUWorker
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies
from .policy_generator import generate_term_life_portfolio
from .portfolio_io import load_term_life_portfolio, write_term_life_portfolio_csv

__all__ = [
    "ActuarialAssumptions",
    "ActuarialProjectionResult",
    "DEFAULT_GPU_THRESHOLD",
    "DEFAULT_STAGE3_GPU_THRESHOLD",
    "GpuTermLifePolicies",
    "TermLifePolicies",
    "WarmGPUWorker",
    "generate_term_life_portfolio",
    "load_term_life_portfolio",
    "project",
    "project_term_life",
    "project_term_life_auto",
    "prepare_gpu_term_life_policies",
    "project_term_life_gpu",
    "project_term_life_gpu_prepared",
    "write_term_life_portfolio_csv",
]
