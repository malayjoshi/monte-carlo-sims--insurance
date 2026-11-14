"""Synthetic fixtures for report validation only, never benchmark evidence."""

import copy
import json

import pytest

from benchmarks import report_stage5 as report


def parity():
    return {"consistent": True, "rtol": 1e-12, "atol": 1e-6, "components": {
        name: {"consistent": True, "max_absolute_difference": 0, "max_relative_difference": 0}
        for name in report.COMPONENTS
    }}


def synthetic_manifest(tmp_path, presets=("small",)):
    manifest = {"schema_version": 1, "execute": True, "status": "succeeded",
                "cwd": str(tmp_path), "runs": []}
    for index, preset in enumerate(presets, 1):
        workload = dict(preset=preset, policies=10 * index, scenarios=10, years=2,
                        seed=3, portfolio_seed=17, policy_chunk=8, scenario_chunk=4,
                        repetitions=2, warmups=1)
        for mode in report.MODES:
            cpu = mode.startswith("cpu_")
            duration = 4 if cpu else 6 / index
            run = {"runtime_seconds": duration}
            if not cpu:
                run.update(parity=parity(), adaptive_gpu={"effective_policy_chunk": 8,
                           "effective_scenario_chunk": 4, "requested_policy_chunk": 8,
                           "requested_scenario_chunk": 4}
                           if mode == "gpu_adaptive" else None)
            summary = {"count": 2, "median_seconds": duration}
            document = {
                "schema_version": 3,
                "workload": {"preset": preset, "policies": workload["policies"], "scenarios": 10,
                             "portfolio_seed": 17, "combinations": 100 * index,
                             "policy_scenario_years": 200 * index,
                             "assumptions": dict(projection_years=2, seed=3,
                                                 policy_chunk_size=8, scenario_chunk_size=4)},
                "environment": {key: "synthetic-test-only" for key in report.ENVIRONMENT_FIELDS},
                "methodology": {},
            }
            document["environment"]["git_commit_sha"] = "a" * 40
            document["environment"]["captured_at_utc"] = mode
            if cpu:
                document.update(portfolio_setup={"source": "generated", "generator_seed": 17},
                                cold_run=run, warm_runs=[run, run], summary=summary)
                document["methodology"] = dict(workers=1 if mode == "cpu_single" else 4,
                                               measured_warm_runs=2, warmup_runs_after_cold=1)
                if mode == "cpu_multicore":
                    document["parity"] = dict(parity(), cold_run=parity(), warm_runs=[parity(), parity()])
            else:
                prepared = dict(run, runtime_seconds=duration / 2)
                document.update(gpu={"name": "synthetic-test-only"}, parity=parity(),
                                cold_compute_run=run, warm_compute_runs=[prepared, prepared],
                                warm_compute_summary=dict(count=2, median_seconds=duration / 2),
                                end_to_end_runs=[run, run], end_to_end_summary=summary)
                document["methodology"] = dict(repetitions=2, warmups=1)
            path = tmp_path / preset / mode / "results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(document))
            manifest["runs"].append({"mode": mode, "workload": workload,
                                     "command": ["python", "-m", "benchmarks.benchmark_stage3_cpu" if cpu
                                                 else "benchmarks.benchmark_stage3_gpu"],
                                     "outputs": {"json_output": str(path)},
                                     "status": "succeeded", "returncode": 0})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def mutate_result(path, mode, change):
    result_path = path.parent / "small" / mode / "results.json"
    document = json.loads(result_path.read_text())
    change(document)
    result_path.write_text(json.dumps(document))


def test_scopes_costs_and_observed_brackets(tmp_path):
    path = synthetic_manifest(tmp_path, ("small", "medium"))
    price = dict(hourly_usd=2, date="2026-01-01", source="synthetic test source only",
                 region="test-region", verified=True)
    result = report.build_report(path, price)
    scopes = result["presets"]["small"]["modes"]["gpu_fixed"]["scopes"]
    assert scopes["warm_host_input"]["speedup_vs_cpu_single"] == pytest.approx(4 / 6)
    assert scopes["warm_prepared_projection"]["speedup_vs_cpu_single"] == pytest.approx(4 / 3)
    assert scopes["warm_host_input"]["shared_host_usd_per_billion_combinations"] == pytest.approx(2 * 6 / 3600 * 1e9 / 100)
    assert len(result["observed_crossover_brackets"]) == 2
    assert all(item["scope"] == "warm_host_input" for item in result["observed_crossover_brackets"])
    text = report.markdown(result)
    assert "not peaks" in text and "complete release" in text and "shared-host" in text


@pytest.mark.parametrize("mode,change", [
    ("gpu_fixed", lambda doc: doc["workload"].update(portfolio_seed=99)),
    ("gpu_fixed", lambda doc: doc["workload"].pop("portfolio_seed")),
    ("cpu_single", lambda doc: doc["portfolio_setup"].update(source="csv")),
    ("gpu_fixed", lambda doc: doc["environment"].update(git_commit_sha="b" * 40)),
    ("cpu_single", lambda doc: doc["environment"].update(git_commit_sha=None)),
    ("cpu_multicore", lambda doc: doc["environment"].update(threading_environment={"OMP_NUM_THREADS": "8"})),
    ("gpu_fixed", lambda doc: doc["warm_compute_runs"].pop()),
    ("gpu_fixed", lambda doc: doc["end_to_end_runs"].pop()),
    ("gpu_fixed", lambda doc: doc["end_to_end_summary"].update(median_seconds=99)),
    ("gpu_fixed", lambda doc: doc["end_to_end_runs"][1]["parity"].update(consistent=False)),
    ("gpu_fixed", lambda doc: doc["warm_compute_runs"][1]["parity"]["components"].pop("claim_pv")),
    ("gpu_adaptive", lambda doc: doc["end_to_end_runs"][1].update(adaptive_gpu=None)),
    ("gpu_adaptive", lambda doc: doc["end_to_end_runs"][1]["adaptive_gpu"].update(requested_policy_chunk=99)),
    ("gpu_fixed", lambda doc: doc["end_to_end_runs"][1]["parity"]["components"]["claim_pv"].update(max_absolute_difference=float("nan"))),
    ("cpu_multicore", lambda doc: doc.update(parity=None)),
    ("cpu_multicore", lambda doc: doc["parity"]["warm_runs"][1].update(consistent=False)),
    ("cpu_multicore", lambda doc: doc["parity"].update(rtol=0.1)),
    ("cpu_multicore", lambda doc: doc["methodology"].update(workers=1)),
    ("gpu_fixed", lambda doc: doc["workload"]["assumptions"].update(seed=4)),
    ("gpu_fixed", lambda doc: doc["workload"]["assumptions"].update(discount_rate=0.9)),
    ("gpu_fixed", lambda doc: doc["end_to_end_runs"][0].update(runtime_seconds=float("nan"))),
    ("gpu_fixed", lambda doc: doc["end_to_end_runs"][0].update(runtime_seconds=-1)),
])
def test_invalid_evidence_rejected(tmp_path, mode, change):
    path = synthetic_manifest(tmp_path)
    mutate_result(path, mode, change)
    with pytest.raises((ValueError, KeyError, TypeError)):
        report.build_report(path)


@pytest.mark.parametrize("change", [
    lambda doc: doc.update(status="dry_run"),
    lambda doc: doc["runs"].pop(),
    lambda doc: doc["runs"].append(copy.deepcopy(doc["runs"][0])),
    lambda doc: doc["runs"][0].update(status="failed"),
    lambda doc: doc["runs"][0]["workload"].update(repetitions=0),
])
def test_invalid_manifest_rejected(tmp_path, change):
    path = synthetic_manifest(tmp_path)
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        report.build_report(path)


def test_downloaded_manifest_paths_are_portable_and_validated(tmp_path):
    path = synthetic_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    for entry in manifest["runs"]:
        entry["outputs"]["json_output"] = f"/opt/synthetic-test-only/small/{entry['mode']}/results.json"
    path.write_text(json.dumps(manifest))
    assert len(report.build_report(path)["presets"]["small"]["modes"]) == 4
    manifest["runs"][0]["outputs"]["json_output"] = "/opt/synthetic-test-only/wrong/results.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="portable"):
        report.build_report(path)


@pytest.mark.parametrize("change", [
    {"hourly_usd": float("nan")}, {"hourly_usd": -1}, {"verified": False},
    {"date": "bad-date"}, {"date": "9999-01-01"}, {"source": " "}, {"region": ""},
])
def test_unverified_or_invalid_pricing_rejected(change):
    price = dict(hourly_usd=1, date="2026-01-01", source="test", region="test", verified=True)
    price.update(change)
    with pytest.raises(ValueError):
        report.validate_price(price)


def test_cli_writes_artifacts_without_inventing_costs(tmp_path):
    path = synthetic_manifest(tmp_path)
    output, summary = tmp_path / "report.md", tmp_path / "summary.json"
    args = [str(path), "--output", str(output), "--json-output", str(summary)]
    assert report.main(args) == 0
    document = json.loads(summary.read_text())
    assert document["price"] is None
    assert "not supplied" in output.read_text()
    assert not document["observed_crossover_brackets"]
    with pytest.raises(SystemExit):
        report.main(args)
    with pytest.raises(SystemExit):
        report.main([*args, "--hourly-price", "1"])


def test_cli_accepts_explicit_verified_dated_price(tmp_path):
    path = synthetic_manifest(tmp_path)
    summary = tmp_path / "summary.json"
    assert report.main([str(path), "--output", str(tmp_path / "report.md"),
                        "--json-output", str(summary), "--hourly-price", "2",
                        "--price-date", "2026-01-01", "--price-source", "synthetic test only",
                        "--price-region", "test", "--price-verified"]) == 0
    assert json.loads(summary.read_text())["price"]["verified"] is True


def test_different_years_do_not_imply_crossover(tmp_path):
    path = synthetic_manifest(tmp_path, ("small", "medium"))
    manifest = json.loads(path.read_text())
    for entry in manifest["runs"][4:]:
        entry["workload"]["years"] = 3
        result_path = tmp_path / "medium" / entry["mode"] / "results.json"
        document = json.loads(result_path.read_text())
        document["workload"]["assumptions"]["projection_years"] = 3
        document["workload"]["policy_scenario_years"] = 600
        result_path.write_text(json.dumps(document))
    path.write_text(json.dumps(manifest))
    assert not report.build_report(path)["observed_crossover_brackets"]
