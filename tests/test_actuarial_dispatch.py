import pytest

import monte_carlo_accelerator.actuarial_dispatch as dispatch
from monte_carlo_accelerator.gpu_engine import GPUUnavailableError
from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio


def test_auto_uses_cpu_below_threshold_without_probing_gpu(monkeypatch):
    policies = generate_term_life_portfolio(10)
    cpu_result = object()
    monkeypatch.setattr(dispatch, "project_term_life", lambda *args: cpu_result)
    monkeypatch.setattr(
        dispatch,
        "project_term_life_gpu",
        lambda *args: pytest.fail("GPU should not be touched"),
    )

    assert dispatch.project_term_life_auto(policies, 9, gpu_threshold=100) is cpu_result


def test_auto_uses_gpu_at_threshold(monkeypatch):
    policies = generate_term_life_portfolio(10)
    gpu_result = object()
    monkeypatch.setattr(dispatch, "project_term_life_gpu", lambda *args: gpu_result)

    assert dispatch.project_term_life_auto(policies, 10, gpu_threshold=100) is gpu_result


def test_auto_falls_back_when_gpu_is_unavailable(monkeypatch):
    policies = generate_term_life_portfolio(10)
    cpu_result = object()
    monkeypatch.setattr(dispatch, "project_term_life", lambda *args: cpu_result)

    def unavailable(*args):
        raise GPUUnavailableError("no CUDA")

    monkeypatch.setattr(dispatch, "project_term_life_gpu", unavailable)

    assert dispatch.project_term_life_auto(policies, 10, gpu_threshold=100) is cpu_result


def test_explicit_gpu_does_not_fall_back(monkeypatch):
    policies = generate_term_life_portfolio(10)

    def unavailable(*args):
        raise GPUUnavailableError("no CUDA")

    monkeypatch.setattr(dispatch, "project_term_life_gpu", unavailable)
    with pytest.raises(GPUUnavailableError, match="no CUDA"):
        dispatch.project_term_life_auto(policies, 1, backend="gpu")


def test_auto_does_not_hide_gpu_execution_errors(monkeypatch):
    policies = generate_term_life_portfolio(10)

    def failed(*args):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(dispatch, "project_term_life_gpu", failed)
    with pytest.raises(RuntimeError, match="kernel failed"):
        dispatch.project_term_life_auto(policies, 10, gpu_threshold=100)


def test_explicit_cpu_ignores_threshold_and_gpu(monkeypatch):
    policies = generate_term_life_portfolio(10)
    cpu_result = object()
    monkeypatch.setattr(dispatch, "project_term_life", lambda *args: cpu_result)
    monkeypatch.setattr(
        dispatch,
        "project_term_life_gpu",
        lambda *args: pytest.fail("GPU should not be touched"),
    )

    assert dispatch.project_term_life_auto(policies, 10, backend="cpu") is cpu_result


def test_dispatch_configuration_is_validated():
    policies = generate_term_life_portfolio(1)
    with pytest.raises(ValueError, match="backend"):
        dispatch.project_term_life_auto(policies, 1, backend="tpu")
    with pytest.raises(ValueError, match="threshold"):
        dispatch.project_term_life_auto(policies, 1, gpu_threshold=0)
