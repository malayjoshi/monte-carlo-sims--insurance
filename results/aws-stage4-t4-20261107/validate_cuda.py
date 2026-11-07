"""Validate Stage 4 on a real CUDA device, including allocator-limit OOM recovery."""

import json
from dataclasses import asdict
from pathlib import Path

import cupy as cp
import numpy as np

from benchmarks.benchmark_stage3_gpu import component_parity, device_metadata
from monte_carlo_accelerator import actuarial_gpu_engine as gpu
from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.models import ActuarialAssumptions
from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio


def main():
    policies = generate_term_life_portfolio(512, seed=17)
    assumptions = ActuarialAssumptions(
        projection_years=3, policy_chunk_size=512, scenario_chunk_size=128
    )
    expected = project_term_life(policies, 128, assumptions)
    prepared = gpu.prepare_gpu_term_life_policies(policies, cp)
    cases = {}
    for name, options in (
        ("default", {}),
        ("restricted_budget", {"memory_fraction": 0.0001}),
    ):
        actual, report = gpu.project_term_life_gpu_adaptive_prepared(
            prepared, 128, assumptions, cp, **options
        )
        parity = component_parity(expected, actual)
        assert parity["consistent"], name
        assert all(
            getattr(actual, component).dtype == np.float64
            for component in ("premium_pv", "claim_pv", "expense_pv")
        )
        if name == "restricted_budget":
            assert report.effective_scenario_chunk < 128
        cases[name] = {"report": asdict(report), "parity": parity}

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    previous_limit = pool.get_limit()
    allocator_limit = 1024 * 1024
    pool.set_limit(size=allocator_limit)
    try:
        actual, report = gpu.project_term_life_gpu_adaptive_prepared(
            prepared, 128, assumptions, cp, max_retries=8
        )
        parity = component_parity(expected, actual)
        assert parity["consistent"]
        assert report.attempts > 1, "allocator limit did not exercise OOM retry"
        cases["allocator_oom_retry"] = {
            "allocator_limit_bytes": allocator_limit,
            "report": asdict(report),
            "parity": parity,
        }
    finally:
        pool.set_limit(size=previous_limit)
        pool.free_all_blocks()

    actual, report = gpu.project_term_life_gpu_adaptive(policies, 128, assumptions)
    parity = component_parity(expected, actual)
    assert parity["consistent"]
    cases["host_api"] = {"report": asdict(report), "parity": parity}
    document = {"gpu": device_metadata(cp), "cases": cases, "passed": True}
    Path("results/cuda_validation.json").write_text(json.dumps(document, indent=2) + "\n")
    print("STAGE4_CUDA_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
