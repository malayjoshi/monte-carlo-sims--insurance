# CPU profiling and benchmark findings

## Scope

This report validates the existing NumPy CPU baseline before GPU implementation. It uses synthetic data and preserves the original model semantics.

## Defensible findings

On an AMD Ryzen 7 5825U (8 cores/16 logical CPUs), the 10-million-combination medium workload produced a 3.984-second warm median across five runs with a 0.100-second sample standard deviation. Median throughput was 2.510 million policy-scenario combinations per second. The first compute invocation was slower at 5.155 seconds, so cold and warm measurements should not be mixed.

A controlled chunk sweep found that performance depends on block shape, not only block element count. The tested `4096 × 128` candidate was confirmed separately with five warmed runs: its 3.502-second median and 2.855-million-combinations/s throughput were about 13.8% better than the original `2048 × 256` setting, with comparable sampled RSS. This is a host-specific tuning result rather than a new universal default.

cProfile attributed 3.060 of 4.068 profiled seconds to the stateless `_uniform` function. That makes deterministic random-number construction the clearest optimization target. Python-visible reductions and mortality-rate calls were much smaller. cProfile cannot break down fused array expressions or native NumPy kernels, so deeper attribution requires native-aware tooling in a later profiling pass.

All tested chunk shapes met `rtol=1e-12, atol=1e-6`. The maximum observed difference was `1.19209289551e-7` absolute and `5.239e-16` relative. Differences arise from floating-point reduction order; random inputs remain independent of chunking.

## Suggested LinkedIn narrative

The useful result is not simply that 500 million combinations fit in memory. Chunking held working memory nearly constant, repeated measurements separated cold from warm behavior, and profiling revealed that deterministic random-number construction—not mortality arithmetic—dominates this implementation. A small block-shape change improved validated medium-workload throughput by about 13.8% on the test machine while preserving numerical results within explicit tolerances.

Avoid presenting the historical large and stress single runs as statistically validated. They remain useful scaling observations, but repeated runs and controlled host conditions are required before quoting them as stable performance figures.

## Next steps

1. Repeat large and stress workloads with the enhanced runner when thermal and power conditions can be controlled.
2. Use a native-aware profiler (for example Linux `perf`, if permitted) to investigate NumPy kernel and memory-bandwidth behavior.
3. Explore reducing temporary arrays in `_uniform` without changing its uint64 semantics.
4. Retest chunk shapes after every substantive engine change and on every target host.
5. Begin GPU implementation only after freezing RNG test vectors and CPU/GPU tolerance tests.
