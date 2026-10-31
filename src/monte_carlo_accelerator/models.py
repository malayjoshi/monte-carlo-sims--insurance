from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Policies:
    policy_id: NDArray[np.uint64]
    age: NDArray[np.int16]
    benefit: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.policy_id)


@dataclass(frozen=True)
class ProjectionConfig:
    years: int = 30
    discount_rate: float = 0.03
    claim_inflation: float = 0.02
    seed: int = 2026
    policy_chunk_size: int = 2_048
    scenario_chunk_size: int = 256


@dataclass(frozen=True)
class ProjectionResult:
    scenario_pv: NDArray[np.float64]

    @property
    def mean_pv(self) -> float:
        return float(np.mean(self.scenario_pv))


@dataclass(frozen=True)
class TermLifePolicies:
    """Validated inputs for a level-premium term-life portfolio."""

    policy_id: NDArray[np.uint64]
    age_at_entry: NDArray[np.int16]
    sum_assured: NDArray[np.float64]
    annual_premium: NDArray[np.float64]
    term_years: NDArray[np.int16]

    def __len__(self) -> int:
        return len(self.policy_id)


@dataclass(frozen=True)
class ActuarialAssumptions:
    """Projection assumptions for the Stage 3 term-life workflow."""

    projection_years: int = 30
    discount_rate: float = 0.03
    mortality_scale: float = 1.0
    annual_lapse_rate: float = 0.05
    expense_per_policy: float = 25.0
    expense_inflation: float = 0.02
    claim_inflation: float = 0.0
    seed: int = 2026
    policy_chunk_size: int = 2_048
    scenario_chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.projection_years <= 0:
            raise ValueError("projection_years must be positive")
        if self.discount_rate <= -1:
            raise ValueError("discount_rate must be greater than -1")
        if self.mortality_scale < 0:
            raise ValueError("mortality_scale must be non-negative")
        if not 0 <= self.annual_lapse_rate <= 1:
            raise ValueError("annual_lapse_rate must be between 0 and 1")
        if self.expense_per_policy < 0:
            raise ValueError("expense_per_policy must be non-negative")
        if self.expense_inflation <= -1 or self.claim_inflation <= -1:
            raise ValueError("inflation assumptions must be greater than -1")
        if self.policy_chunk_size <= 0 or self.scenario_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")


@dataclass(frozen=True)
class ActuarialProjectionResult:
    """Scenario-level present values for the main insurance cash flows."""

    premium_pv: NDArray[np.float64]
    claim_pv: NDArray[np.float64]
    expense_pv: NDArray[np.float64]

    @property
    def net_cashflow_pv(self) -> NDArray[np.float64]:
        return self.premium_pv - self.claim_pv - self.expense_pv

    @property
    def mean_net_cashflow_pv(self) -> float:
        return float(np.mean(self.net_cashflow_pv))
