import pytest

from monte_carlo_accelerator.metrics import Measurement, measure, summarize


def test_measure_returns_value_runtime_and_memory():
    result = measure(lambda value: value + 1, 2, memory_sample_interval=0.001)
    assert result.value == 3
    assert result.runtime_seconds >= 0
    assert result.peak_memory_mb >= 0


def test_summary_statistics_and_throughput():
    values = [Measurement(None, runtime, 10.0) for runtime in (1.0, 2.0, 3.0)]
    summary = summarize(values, 12)
    assert summary.median_seconds == 2.0
    assert summary.mean_seconds == 2.0
    assert summary.min_throughput == 4.0
    assert summary.max_throughput == 12.0


def test_summary_requires_a_measurement():
    with pytest.raises(ValueError):
        summarize([], 1)
