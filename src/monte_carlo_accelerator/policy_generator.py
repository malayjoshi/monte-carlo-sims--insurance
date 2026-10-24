import numpy as np

from .models import Policies


def generate_policies(count: int, seed: int = 17) -> Policies:
    """Generate a reproducible synthetic term-life portfolio."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return Policies(
        policy_id=np.arange(count, dtype=np.uint64),
        age=rng.integers(20, 71, count, dtype=np.int16),
        benefit=np.round(rng.lognormal(np.log(100_000), 0.55, count), 2),
    )
