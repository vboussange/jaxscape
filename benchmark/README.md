## Benchmark workspace

This workspace owns the cross-tool benchmarks that feed the documentation site.
Each run writes structured results to `benchmark/results/benchmark_results.json`
and `benchmark/results/benchmark_results.csv`, then renders one scorecard per
feature into `docs/assets/`. Every automated feature is benchmarked on three
deterministic size tiers: `small`, `medium`, and `large`.

## Compatibility summary

| Feature | Automated profiles | Optional local profiles | Notes |
| --- | --- | --- | --- |
| Resistance distance | `JAXScape / pinv / f32 (CPU/GPU)`, `JAXScape / pinv / f64 (CPU/GPU)`, `JAXScape / approx pinv / f32 (CPU/GPU)`, `JAXScape / approx pinv / f64 (CPU/GPU)`, `JAXScape / PyAMG`, `gdistance / commuteDistance`, `Circuitscape.jl / cg+amg`, `Circuitscape.jl / cholmod` | `JAXScape / AMJaxCGSolver / f32 (CPU/GPU)`, `JAXScape / AMJaxCGSolver / f64 (CPU/GPU)`, `JAXScape / CholmodSolver`, `JAXScape / approx CholmodSolver`, `Conefor` adapters | Use `--device cpu` for cross-tool comparisons. The GPU series is emitted for GPU-capable JAXScape methods. |
| Least-cost path | `JAXScape (CPU/GPU)`, `gdistance / costDistance` | `Conefor` adapter | Conefor remains a manual external integration. |
| Sensitivity analysis | `JAXScape / shortest-path gradient (CPU/GPU)`, `gdistance / shortestPath`, `JAXScape / resistance gradient (CPU/GPU)`, `gdistance / passage` | none | This scorecard compares JAX gradients to the matching `gdistance` centrality surfaces. |
| Inverse landscape genetics | `JAXScape + Optimistix (CPU/GPU)`, `ResistanceGA` | none | Both adapters run a fixed-budget synthetic optimisation problem for regression checks. |

## Tasks

The exact mathematical setup of the raster generator, point sets, and objective
functions is documented on `docs/benchmark.md` and mirrored into the generated
`benchmark_results.json` case metadata.

Every run also records the normalized thread configuration in the JSON artifact,
including `BENCHMARK_THREADS`, the propagated BLAS/OpenMP and R/Julia thread
variables, and the `XLA_FLAGS` used to pin CPU-side JAX execution. The workspace
default, and the published CI default, is `BENCHMARK_THREADS=4`.

## Workspace layout

- `benchmark/benchmark_distances.py`: full-suite orchestration, shared synthetic cases, cross-software adapters, and JSON/CSV artifact writing.
- `benchmark/jaxscape/`: JAXScape-only task entry points plus `utils.py`, which owns the shared standalone runner, worker subprocess plumbing, and task-level benchmark helpers.
- `benchmark/render_scorecard.py` and `benchmark/run_benchmarks.sh`: artifact rendering and end-to-end reruns.
- `benchmark/install_external_tools.sh`, `benchmark/julia/`, `benchmark/.julia/`, and `benchmark/.r-lib/`: local Julia and R benchmark toolchain setup.
- `benchmark/external/`: adapter scripts for Circuitscape, `gdistance`, Conefor, and `ResistanceGA`.

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
Benchmark entry points set `XLA_PYTHON_CLIENT_PREALLOCATE=false` by default,
unless the environment already defines a different value, so the coordinator
does not reserve most GPU memory before the worker subprocesses start.