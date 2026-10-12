import os
import resource
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Measurement:
    value: object
    runtime_seconds: float
    peak_memory_mb: float


def measure(function: Callable, *args, **kwargs) -> Measurement:
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    value = function(*args, **kwargs)
    runtime = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if os.name != "darwin" else 1024.0 * 1024.0
    return Measurement(value, runtime, peak_rss / divisor)
