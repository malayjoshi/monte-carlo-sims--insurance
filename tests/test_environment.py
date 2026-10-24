from monte_carlo_accelerator.environment import collect_environment


def test_environment_contains_reproducibility_fields():
    metadata = collect_environment()
    expected = {
        "captured_at_utc", "cpu_model", "physical_cpu_cores", "logical_cpu_count", "total_ram",
        "linux_distribution", "kernel", "python_version", "numpy_version",
        "blas_configuration", "threading_environment", "git_commit_sha",
    }
    assert expected <= metadata.keys()
