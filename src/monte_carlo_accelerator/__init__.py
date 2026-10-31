"""Monte Carlo accelerator."""

from .actuarial_engine import project_term_life
from .engine import DEFAULT_GPU_THRESHOLD, project
from .gpu_worker import WarmGPUWorker
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies
from .portfolio_io import load_term_life_portfolio

__all__ = [
    "ActuarialAssumptions",
    "ActuarialProjectionResult",
    "DEFAULT_GPU_THRESHOLD",
    "TermLifePolicies",
    "WarmGPUWorker",
    "load_term_life_portfolio",
    "project",
    "project_term_life",
]
