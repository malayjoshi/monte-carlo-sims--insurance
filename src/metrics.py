import statistics
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Measurement:
    value: object
    runtime_seconds: float
    peak_memory_mb: float


@dataclass(frozen=True)
class RuntimeSummary:
    count: int
    mean_seconds: float
    median_seconds: float
    stdev_seconds: float
    min_seconds: float
    max_seconds: float
    mean_throughput: float
    median_throughput: float
    min_throughput: float
    max_throughput: float


def _rss_mb() -> float:
    """Return current resident memory on Linux, or zero when unavailable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def measure(function: Callable, *args, memory_sample_interval: float = 0.005, **kwargs) -> Measurement:
    """Time one call and sample its process RSS (setup must happen beforehand)."""
    stop = threading.Event()
    samples = [_rss_mb()]

    def sample_memory() -> None:
        while not stop.wait(memory_sample_interval):
            samples.append(_rss_mb())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        value = function(*args, **kwargs)
    finally:
        runtime = time.perf_counter() - start
        stop.set()
        sampler.join()
        samples.append(_rss_mb())
    return Measurement(value, runtime, max(samples))


def summarize(measurements: list[Measurement], work_items: int) -> RuntimeSummary:
    if not measurements:
        raise ValueError("at least one measurement is required")
    runtimes = [item.runtime_seconds for item in measurements]
    throughputs = [work_items / runtime for runtime in runtimes]
    return RuntimeSummary(
        count=len(runtimes),
        mean_seconds=statistics.fmean(runtimes),
        median_seconds=statistics.median(runtimes),
        stdev_seconds=statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
        min_seconds=min(runtimes),
        max_seconds=max(runtimes),
        mean_throughput=statistics.fmean(throughputs),
        median_throughput=statistics.median(throughputs),
        min_throughput=min(throughputs),
        max_throughput=max(throughputs),
    )
