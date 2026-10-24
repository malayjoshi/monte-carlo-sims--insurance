import pytest

import src.engine as engine
from src.gpu_engine import GPUUnavailableError
from src.models import ProjectionResult
from src.policy_generator import generate_policies


def _result(tag):
    return ProjectionResult(tag)


def test_auto_uses_cpu_below_threshold_without_probing_gpu(monkeypatch):
    policies = generate_policies(10)
    monkeypatch.setattr(engine, "project_cpu", lambda *args: _result("cpu"))
    monkeypatch.setattr(engine, "project_gpu", lambda *args: pytest.fail("GPU should not be touched"))

    assert engine.project(policies, 9, gpu_threshold=100).scenario_pv == "cpu"


def test_auto_uses_gpu_at_threshold(monkeypatch):
    policies = generate_policies(10)
    monkeypatch.setattr(engine, "project_cpu", lambda *args: _result("cpu"))
    monkeypatch.setattr(engine, "project_gpu", lambda *args: _result("gpu"))

    assert engine.project(policies, 10, gpu_threshold=100).scenario_pv == "gpu"


def test_auto_falls_back_only_when_gpu_is_unavailable(monkeypatch):
    policies = generate_policies(10)
    monkeypatch.setattr(engine, "project_cpu", lambda *args: _result("cpu"))

    def unavailable(*args):
        raise GPUUnavailableError("no CUDA")

    monkeypatch.setattr(engine, "project_gpu", unavailable)
    assert engine.project(policies, 10, gpu_threshold=100).scenario_pv == "cpu"


def test_explicit_gpu_does_not_fall_back(monkeypatch):
    policies = generate_policies(10)

    def unavailable(*args):
        raise GPUUnavailableError("no CUDA")

    monkeypatch.setattr(engine, "project_gpu", unavailable)
    with pytest.raises(GPUUnavailableError, match="no CUDA"):
        engine.project(policies, 1, backend="gpu")


def test_gpu_worker_is_reused_by_dispatch(monkeypatch):
    policies = generate_policies(10)

    class Worker:
        def project(self, *args):
            return _result("warm-gpu")

    monkeypatch.setattr(engine, "project_gpu", lambda *args: pytest.fail("cold GPU path used"))
    result = engine.project(policies, 10, gpu_threshold=100, gpu_worker=Worker())
    assert result.scenario_pv == "warm-gpu"


def test_dispatch_configuration_is_validated():
    policies = generate_policies(1)
    with pytest.raises(ValueError, match="backend"):
        engine.project(policies, 1, backend="tpu")
    with pytest.raises(ValueError, match="threshold"):
        engine.project(policies, 1, gpu_threshold=0)
