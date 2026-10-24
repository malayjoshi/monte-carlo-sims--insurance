import numpy as np

import src.gpu_worker as gpu_worker
from src.gpu_engine import GpuPolicies
from src.models import ProjectionResult
from src.policy_generator import generate_policies


def test_worker_loads_backend_once_and_warms_once(monkeypatch):
    backend = object()
    calls = {"load": 0, "prepare": 0, "project": 0}

    def load():
        calls["load"] += 1
        return backend

    def prepare(policies, cupy):
        assert cupy is backend
        calls["prepare"] += 1
        return GpuPolicies(policies.policy_id, policies.age, policies.benefit)

    def project(prepared, scenarios, config, cupy):
        assert cupy is backend
        calls["project"] += 1
        return ProjectionResult(np.zeros(scenarios))

    monkeypatch.setattr(gpu_worker, "_load_cupy", load)
    monkeypatch.setattr(gpu_worker, "prepare_gpu_policies", prepare)
    monkeypatch.setattr(gpu_worker, "project_gpu_prepared", project)

    worker = gpu_worker.WarmGPUWorker()
    worker.warmup()
    worker.project(generate_policies(2), 3)

    assert worker.warmed
    assert calls == {"load": 1, "prepare": 2, "project": 2}


def test_worker_can_reuse_prepared_policies(monkeypatch):
    backend = object()
    prepared = GpuPolicies(np.array([1]), np.array([40]), np.array([100_000.0]))
    calls = []

    def project(value, scenarios, config, cupy):
        calls.append((value, scenarios, cupy))
        return ProjectionResult(np.zeros(scenarios))

    monkeypatch.setattr(gpu_worker, "project_gpu_prepared", project)
    worker = gpu_worker.WarmGPUWorker(warmup=False, cupy=backend)

    result = worker.project_prepared(prepared, 4)

    assert result.scenario_pv.shape == (4,)
    assert calls == [(prepared, 4, backend)]
