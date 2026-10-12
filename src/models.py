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
