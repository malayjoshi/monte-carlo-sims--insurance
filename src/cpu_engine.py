import numpy as np

from .models import Policies, ProjectionConfig, ProjectionResult

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def _uniform(policy_id, scenario_id, year: int, seed: int):
    """Stateless SplitMix64 uniforms indexed by policy, scenario, and year."""
    with np.errstate(over="ignore"):
        x = (np.asarray(policy_id, dtype=np.uint64) * _C1) & _MASK
        x = x ^ ((np.asarray(scenario_id, dtype=np.uint64) * _C2) & _MASK)
        x = x ^ ((np.uint64(year + 1) * _C3) & _MASK)
        x = x ^ np.uint64(seed)
        x = (x + _C1) & _MASK
        x = ((x ^ (x >> np.uint64(30))) * _C2) & _MASK
        x = ((x ^ (x >> np.uint64(27))) * _C3) & _MASK
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def _mortality_rate(age):
    """Smooth illustrative annual mortality curve, capped at 35%."""
    return np.minimum(0.00035 * np.exp(0.085 * (np.asarray(age) - 30)), 0.35)


def project_cpu(policies: Policies, scenarios: int, config: ProjectionConfig = ProjectionConfig()) -> ProjectionResult:
    """Vectorized, chunked projection returning aggregate claim PV by scenario."""
    if scenarios <= 0 or config.years <= 0:
        raise ValueError("scenarios and years must be positive")
    totals = np.zeros(scenarios, dtype=np.float64)
    for p0 in range(0, len(policies), config.policy_chunk_size):
        p1 = min(p0 + config.policy_chunk_size, len(policies))
        ids = policies.policy_id[p0:p1, None]
        ages = policies.age[p0:p1, None]
        benefits = policies.benefit[p0:p1, None]
        for s0 in range(0, scenarios, config.scenario_chunk_size):
            s1 = min(s0 + config.scenario_chunk_size, scenarios)
            scenario_ids = np.arange(s0, s1, dtype=np.uint64)[None, :]
            alive = np.ones((p1 - p0, s1 - s0), dtype=bool)
            block_pv = np.zeros(s1 - s0, dtype=np.float64)
            for year in range(config.years):
                deaths = alive & (_uniform(ids, scenario_ids, year, config.seed) < _mortality_rate(ages + year))
                factor = (1.0 + config.claim_inflation) ** year / (1.0 + config.discount_rate) ** (year + 1)
                block_pv += np.sum(deaths * benefits, axis=0) * factor
                alive &= ~deaths
            totals[s0:s1] += block_pv
    return ProjectionResult(totals)


def project_reference(policies: Policies, scenarios: int, config: ProjectionConfig = ProjectionConfig()) -> ProjectionResult:
    """Deliberately simple scalar-loop oracle for small correctness tests."""
    totals = np.zeros(scenarios, dtype=np.float64)
    for p in range(len(policies)):
        for s in range(scenarios):
            for year in range(config.years):
                qx = float(_mortality_rate(int(policies.age[p]) + year))
                u = float(_uniform(policies.policy_id[p], np.uint64(s), year, config.seed))
                if u < qx:
                    factor = (1.0 + config.claim_inflation) ** year / (1.0 + config.discount_rate) ** (year + 1)
                    totals[s] += policies.benefit[p] * factor
                    break
    return ProjectionResult(totals)
