"""Long-lived GPU worker that keeps CUDA initialized between projections."""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from .gpu_engine import (
    GpuPolicies,
    _load_cupy,
    prepare_gpu_policies,
    project_gpu_prepared,
)
from .models import Policies, ProjectionConfig, ProjectionResult


class WarmGPUWorker:
    """Own one warm CUDA context and serialize projections through it.

    Construct this object once per process and reuse it for multiple requests.
    Initialization performs a minimal projection so CUDA context creation and
    CuPy kernel compilation do not affect the first real request.
    """

    def __init__(self, *, warmup: bool = True, cupy: Any | None = None) -> None:
        self._cupy = cupy if cupy is not None else _load_cupy()
        self._lock = Lock()
        self._warmed = False
        if warmup:
            self.warmup()

    @property
    def warmed(self) -> bool:
        """Whether the one-time representative warm-up has completed."""
        return self._warmed

    def warmup(self) -> None:
        """Initialize CUDA and compile the kernels used by real projections."""
        with self._lock:
            if self._warmed:
                return
            policies = Policies(
                policy_id=np.array([0], dtype=np.uint64),
                age=np.array([40], dtype=np.int16),
                benefit=np.array([100_000.0], dtype=np.float64),
            )
            prepared = prepare_gpu_policies(policies, self._cupy)
            config = ProjectionConfig(years=1, policy_chunk_size=1, scenario_chunk_size=1)
            project_gpu_prepared(prepared, 1, config, self._cupy)
            self._warmed = True

    def prepare(self, policies: Policies) -> GpuPolicies:
        """Copy policies to the worker's GPU for reuse across projections."""
        with self._lock:
            return prepare_gpu_policies(policies, self._cupy)

    def project(
        self,
        policies: Policies,
        scenarios: int,
        config: ProjectionConfig = ProjectionConfig(),
    ) -> ProjectionResult:
        """Transfer policies and run a projection on the warm GPU."""
        with self._lock:
            prepared = prepare_gpu_policies(policies, self._cupy)
            return project_gpu_prepared(prepared, scenarios, config, self._cupy)

    def project_prepared(
        self,
        policies: GpuPolicies,
        scenarios: int,
        config: ProjectionConfig = ProjectionConfig(),
    ) -> ProjectionResult:
        """Run with policies already resident on this worker's GPU."""
        with self._lock:
            return project_gpu_prepared(policies, scenarios, config, self._cupy)
