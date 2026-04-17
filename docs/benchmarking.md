# Benchmarking

JAXScape now ships with a small benchmark workspace inspired by the structure of
SciMLBenchmarks: one orchestrator, tool-specific runners, committed result
artifacts, and a rendered scorecard used by the documentation.

## Workspace layout

- `benchmark/benchmark_distances.py`: orchestrates the benchmark suite and writes JSON/CSV results
- `benchmark/run_benchmarks.sh`: reruns the suite and regenerates the scorecard image
- `benchmark/install_external_tools.sh`: installs `Circuitscape.jl` and documents the extra tooling needed for `Conefor`, `samc`, and `ResistanceGA`
- `benchmark/external/circuitscape_resistance.jl`: Julia runner for the Circuitscape resistance benchmark
- `benchmark/external/conefor_runner.sh`: Conefor adapter hook
- `benchmark/external/samc_sensitivity.R`: `samc` adapter hook
- `benchmark/external/resistancega_inverse.R`: `ResistanceGA` adapter hook
- `benchmark/results/benchmark_results.json`: committed benchmark runs
- `benchmark/results/benchmark_results.csv`: flattened benchmark table

## Current scorecard

<div align="center"><img src="assets/benchmark_scorecard.png" alt="Benchmark scorecard" width="900"></div>

The committed scorecard is generated from `benchmark/results/benchmark_results.json`.
Measured cells are real timings collected in this repository; grey cells mark
adapters that are wired into the orchestrator but were unavailable in the local
benchmark environment.

## Regenerating the artifacts

```bash
./benchmark/install_external_tools.sh   # optional
./benchmark/run_benchmarks.sh
```

The current suite covers:

- `JAXScape` resistance distance
- `Circuitscape.jl` resistance distance
- `Conefor` placeholders for resistance and least-cost-path comparisons
- `JAXScape` least-cost-path distance
- `JAXScape` sensitivity analysis
- `samc` sensitivity adapter hook
- `JAXScape + Optimistix` inverse landscape genetics
- `ResistanceGA` inverse adapter hook

This keeps the benchmark workflow outside `/tests` while still committing the
produced scorecard into the documentation site.
