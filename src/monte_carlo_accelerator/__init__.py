"""Monte Carlo accelerator."""

from .actuarial_engine import project_term_life
from .engine import DEFAULT_GPU_THRESHOLD, project
from .gpu_worker import WarmGPUWorker
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies
from .policy_generator import generate_term_life_portfolio
from .portfolio_io import load_term_life_portfolio, write_term_life_portfolio_csv

__all__ = [
    "ActuarialAssumptions",
    "ActuarialProjectionResult",
    "DEFAULT_GPU_THRESHOLD",
    "TermLifePolicies",
    "WarmGPUWorker",
    "generate_term_life_portfolio",
    "load_term_life_portfolio",
    "project",
    "project_term_life",
    "write_term_life_portfolio_csv",
]
