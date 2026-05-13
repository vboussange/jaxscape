## Benchmark workspace

This workspace owns the cross-tool benchmarks that feed the documentation site.
Each run writes structured results to `benchmark/results/benchmark_results.json`
and `benchmark/results/benchmark_results.csv`, then renders one scorecard per
feature into `docs/assets/`. Every automated feature is benchmarked on three
deterministic size tiers: `small`, `medium`, and `large`.

## Compatibility summary

| Feature | Automated profiles | Optional local profiles | Notes |
| --- | --- | --- | --- |
| Resistance distance | `JAXScape / pinv (CPU/GPU)`, `JAXScape / PyAMG`, `gdistance / commuteDistance`, `Circuitscape.jl / cg+amg`, `Circuitscape.jl / cholmod` | `JAXScape / AMJaxCGSolver`, `JAXScape / CholmodSolver`, `Conefor` adapters | Use `--device cpu` for cross-tool comparisons. The GPU series is only emitted for GPU-capable JAXScape methods. |
| Least-cost path | `JAXScape (CPU/GPU)`, `gdistance / costDistance` | `Conefor` adapter | Conefor remains a manual external integration. |
| Sensitivity analysis | `JAXScape / shortest-path gradient (CPU/GPU)`, `gdistance / shortestPath`, `JAXScape / resistance gradient (CPU/GPU)`, `gdistance / passage` | none | This scorecard compares JAX gradients to the matching `gdistance` centrality surfaces. |
| Inverse landscape genetics | `JAXScape + Optimistix (CPU/GPU)`, `ResistanceGA` | none | Both adapters run a fixed-budget synthetic optimisation problem for regression checks. |

## Problem sizes

| Feature | Small | Medium | Large |
| --- | --- | --- | --- |
| Resistance distance | `12 x 12`, seed `0` | `18 x 18`, seed `1` | `24 x 24`, seed `2` |
| Least-cost path | `16 x 16`, seed `10` | `24 x 24`, seed `11` | `32 x 32`, seed `12` |
| Sensitivity analysis | `6 x 6`, seed `20` | `8 x 8`, seed `21` | `10 x 10`, seed `22` |
| Inverse landscape genetics | `6 x 6`, seed `30` | `8 x 8`, seed `31` | `10 x 10`, seed `32` |

The exact mathematical setup of the raster generator, point sets, and objective
functions is documented on `docs/benchmark.md` and mirrored into the generated
`benchmark_results.json` case metadata.

Every run also records the normalized thread configuration in the JSON artifact,
including `BENCHMARK_THREADS`, the propagated BLAS/OpenMP and R/Julia thread
variables, and the `XLA_FLAGS` used to pin CPU-side JAX execution. The workspace
default, and the published CI default, is `BENCHMARK_THREADS=4`.

## Workspace layout

- `benchmark/benchmark_distances.py`: owns the full-suite orchestration, cross-software adapters, shared cases, result writing, and JSON/CSV output.
- `benchmark/jaxscape/`: JAXScape-only benchmark task modules imported by the orchestrator; each has a lightweight no-argument `main()` for standalone execution.
- `benchmark/jaxscape/resistance_distance.py`: resistance-distance task runner. The JAXScape solver profiles live in `JAXSCAPE_RESISTANCE_PROFILES`, so adding a new solver should usually mean adding one registry entry and a small factory.
- `benchmark/jaxscape/least_cost_path.py`: least-cost path task runner.
- `benchmark/jaxscape/sensitivity_analysis.py`: gradient and centrality sensitivity task runner.
- `benchmark/jaxscape/inverse_landscape_genetics.py`: inverse landscape genetics task runner.
- `benchmark/render_scorecard.py`: renders one PNG scorecard per feature from the JSON results.
- `benchmark/run_benchmarks.sh`: reruns the suite and regenerates the scorecards.
- `benchmark/install_external_tools.sh`: provisions local Julia and R toolchains under `benchmark/julia`, `benchmark/.julia`, and `benchmark/.r-lib`.
- `benchmark/external/circuitscape_resistance.jl`: Julia runner for the Circuitscape resistance profiles.
- `benchmark/external/gdistance_runner.R`: R runner for `gdistance` least-cost, commute-distance, shortest-path, and passage-centrality profiles.
- `benchmark/external/conefor_runner.sh`: Conefor adapter hook.
- `benchmark/external/resistancega_inverse.R`: `ResistanceGA` inverse-landscape-genetics adapter.

## Regenerating the artifacts

```bash
uv sync --extra benchmark
./benchmark/install_external_tools.sh
./benchmark/run_benchmarks.sh --device cpu --require-complete
```

For local GPU profiling, keep the same workflow and switch the device flag:

```bash
./benchmark/run_benchmarks.sh --device gpu
```

To include the optional direct JAXScape solvers locally, install the extras once:

```bash
uv sync --extra benchmark --extra cholespy
```

To benchmark `AMJaxCGSolver`, also enable the `amjax` extra:

```bash
uv sync --extra benchmark --extra amjax
```

The modules under `benchmark/jaxscape/` can be executed directly without CLI
flags. For example:

```bash
uv run --extra benchmark python benchmark/jaxscape/resistance_distance.py
```

Those lightweight entry points use the default benchmark configuration and keep
JAXScape-specific benchmark code separate from the cross-software orchestration.