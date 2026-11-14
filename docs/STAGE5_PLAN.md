# Stage 5: matched validation and release

Status: local orchestration and timing/parity updates implemented. Actual Stage 5 AWS validation, final report and release remain pending.

## Goal and completion criteria

Publish a reproducible comparison of single-worker CPU, multicore CPU, fixed-chunk GPU, and adaptive GPU for the same synthetic term-life model. Finish with passing CPU CI, recorded CUDA validation, a benchmark report, and a tagged release.

Stage 4 is already merged into GitHub main at `1cf54f8646fc0ebf66498c53e681afaed7537b61`. Its CUDA archive establishes correctness and tested capacity through 500 million combinations; it does not establish a universal speedup or maximum capacity. Preserve the local README and untracked tracker edits when updating the checkout.

## Work assignments

| Owner | Deliverable | Acceptance |
|---|---|---|
| Benchmark subagent | Minimal Stage 5 command runner and focused tests | Reuses existing runners; identical workload flags; separate artifacts; dry-run default; failures stop execution |
| Architecture subagent | `ARCHITECTURE.md` with Mermaid diagram | Shows actual APIs, data flow, GPU memory decisions, outputs, and timing limits |
| Main agent | This plan, CPU CI, integration review | Scope and measurement boundaries explicit; local tests pass |
| Implemented measurement updates | Schema 3 CPU/GPU runners | Repeated adaptive and end-to-end timings; every measured result checked for component parity |
| AWS validation run | Matched measurements and environment archive | All four modes run on the same pinned source and documented hardware |
| Release closeout | Final report and release | Evidence supports each claim; CI and CUDA validation pass |

Use narrow file ownership and short agent summaries. Do not duplicate engine implementations or add a benchmark framework. The architecture remains a Python package plus command-line tools; no service, database, or dashboard is required.

## Implemented runner and measurement boundaries

[benchmark_stage5.py](../benchmarks/benchmark_stage5.py) now orchestrates all four modes with matching workload flags and requested chunks. It defaults to preview, writing a schema 1 `manifest.json` with commands and artifact paths. `--execute` runs the existing Stage 3 runners sequentially, records status and stops on failure, marking remaining runs skipped. It requires a fresh or empty output directory and separates artifacts by preset/mode. It does not provision AWS resources.

The updated Stage 3 runners emit schema 3 JSON:

1. Prepared GPU compute still includes copying three scenario arrays to the host; it is not kernel-only timing.
2. Repeated host-facing GPU calls populate `end_to_end_runs` and `end_to_end_summary` with median and spread. CUDA is already initialized; process startup and portfolio generation are excluded. Legacy `end_to_end_run` retains the first sample, not the median.
3. Every adaptive measured call, including cold, warm prepared and end-to-end calls, reruns selection and the retry wrapper. Timings include that overhead and any retries. Warmups also use adaptive execution. Per-run reports are authoritative; top-level adaptive metadata describes the cold prepared call.
4. Every measured GPU result and every CPU cold/warm result checks premium, claim, expense and net PV at `rtol=1e-12, atol=1e-6`. GPU and multicore use untimed single-worker references; single-worker mode uses its first warm result as a self-reference. CPU exact warm repeatability remains checked. Failures prevent successful artifact writes. GPU top-level parity retains the first warm check; per-run records provide full coverage.
5. CPU RSS and GPU allocator/device snapshots remain distinct. GPU snapshots are not peak device-memory measurements. Add peak sampling only if making peak-memory claims.

These implementation changes close the prior timing/parity coverage gaps, not the requirement for actual matched CUDA measurements and retained evidence.

## Benchmark protocol

- Pin one integrated source commit and record Python, NumPy, CuPy, CUDA, GPU, CPU topology, OS, worker count, and thread settings.
- Use the same host for the initial four-way comparison, reusing the Stage 4 `g4dn.xlarge`/Tesla T4 setup where available. A separate CPU-only host comparison is optional and must be labeled separately.
- Use generated portfolios with portfolio seed 17, projection seed 2026, 30 years, and requested chunks 4096 by 128. Keep all actuarial assumptions identical.
- Use one warmup and five measured repetitions per configuration. Record cold invocation and CUDA initialization separately. Report median, spread, throughput, requested/effective chunks, and retries.
- Start with small and medium workloads. Run large and stress after correctness passes. Choose multicore workers from measured topology, recording the exact value.
- For crossover, add workload points around the observed change in winner using the existing runners' `--policies` and `--scenarios` options. Report a tested interval or no observed crossover, rather than inventing an exact threshold.
- Archive JSON, CSV, commands, logs, source identity, test output, environment, and checksums in a unique run directory. Repeat with reversed mode order if differences are small or host drift is evident.

## Analysis and release

Compute speedup as single-worker CPU median divided by the compared median, using the same timing scope. Compute cost per billion combinations as `(hourly price / 3600) * seconds * (1e9 / combinations)`. Record price source, date, region, currency, purchase model, and exclusions. Do not treat shared-host timings as a comparison against a differently priced CPU instance.

The final report must include workload definitions, architecture diagram, methodology, raw artifact links, numerical tolerances, performance and cost tables, uncertainty, and synthetic-data limitations. Distinguish warm throughput from interactive cold-start latency. Do not call 500 million combinations a maximum.

Run CPU CI and CUDA checks on the final integrated source. Archive results before terminating validation resources. Publish the release only after all acceptance items pass; do not mark Stage 5 complete from scaffolding or local CPU tests alone.

## First local check

```bash
python -m pytest -q
python benchmarks/benchmark_stage5.py --help
```

The Stage 5 runner defaults to command preview. GPU execution and paid AWS provisioning are separate from preparing this plan.
