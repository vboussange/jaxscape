## Benchmark workspace

This workspace owns the cross-tool benchmarks that feed the documentation site.
The default full-suite entrypoint writes structured results to
`benchmark/results/benchmark_results.json` and
`benchmark/results/benchmark_results.csv`, then renders one scorecard per
feature into `docs/assets/`. Long GPU launches can also be run through the
run-scoped wrapper at `benchmark/results/runs/<timestamp>/`, and the smoke
pipeline writes isolated artifacts under `benchmark/results/smoke/`. Every
automated feature is benchmarked on deterministic synthetic size tiers; the
sensitivity suite uses a smaller `large` case than the distance tasks.

## Compatibility summary

| Feature | Automated profiles | Optional local profiles | Notes |
| --- | --- | --- | --- |
| Resistance distance | `JAXScape / PyAMG`, `gdistance / commuteDistance`, `Circuitscape.jl / cg+amg / f64`, `Circuitscape.jl / cholmod / f64` | `JAXScape / CholmodSolver / f32`, `JAXScape / CholmodSolver / f64`, `JAXScape / AMJaxCGSolver / f32 (CPU/GPU)`, `JAXScape / AMJaxCGSolver / f64 (CPU/GPU)`, `JAXScape / approx AMJaxCGSolver / f32 (CPU/GPU)`, `JAXScape / approx AMJaxCGSolver / f64 (CPU/GPU)`, `JAXScape / approx CholmodSolver / f32`, `JAXScape / approx CholmodSolver / f64`, `Conefor` adapters | The dense `pinv` family is intentionally excluded from the full resistance suite because the large resistance case is not a useful dense-pseudoinverse benchmark target. |
| Least-cost path | `JAXScape (CPU/GPU)`, `gdistance / costDistance` | `Conefor` adapter | Conefor remains a manual external integration. |
| Sensitivity analysis | `JAXScape / shortest-path gradient (CPU/GPU)`, `gdistance / shortestPath`, `JAXScape / resistance gradient (CPU/GPU)`, `gdistance / passage` | none | This scorecard compares JAX gradients to the matching `gdistance` centrality surfaces. |
| Inverse landscape genetics | `JAXScape / CholmodSolver / f32`, `JAXScape / AMJaxCGSolver / f32 (CPU/GPU)`, `JAXScape / AMJaxCGSolver / f64 (CPU/GPU)`, `JAXScape / approx pinv / f32 (CPU/GPU)`, `JAXScape / approx pinv / f64 (CPU/GPU)`, `ResistanceGA` | none | JAXScape runs the fixed-budget synthetic inverse problem through Optimistix LBFGS and reports final fit quality on a shared commute-distance evaluation metric. |

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
- `benchmark/render_scorecard.py`, `benchmark/generate_benchmark_docs.py`, and `benchmark/run_benchmarks.sh`: scorecard rendering, docs generation, and end-to-end reruns into the canonical published paths.
- `benchmark/run_smoke_checks.sh`: representative smoke pipeline that writes isolated JSON/CSV, scorecards, and a generated benchmark markdown page under `benchmark/results/smoke/`.
- `benchmark/launch_full_benchmark.sh`: asynchronous run-scoped launcher that writes logs, scorecards, generated docs, and run status under `benchmark/results/runs/<timestamp>/`.
- `benchmark/publish_benchmark_artifacts.py`: promote a validated run-scoped artifact set back into the canonical `benchmark/results/` and `docs/assets/` paths.
- `benchmark/install_external_tools.sh`, `benchmark/julia/`, `benchmark/.julia/`, and `benchmark/.r-lib/`: local Julia and R benchmark toolchain setup.
- `benchmark/external/`: adapter scripts for Circuitscape, `gdistance`, Conefor, and `ResistanceGA`.

## Regenerating the artifacts

```bash
uv sync --extra benchmark --extra amjax --extra cholespy
./benchmark/install_external_tools.sh
./benchmark/run_benchmarks.sh --device cpu --require-complete
```

For local GPU profiling, install CUDA support and switch the device flag:

```bash
uv sync --extra benchmark --extra amjax --extra cholespy --extra cuda
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

To run the representative smoke benchmark and render isolated smoke artifacts:

```bash
./benchmark/run_smoke_checks.sh --require-complete
```

To launch the full benchmark asynchronously into a run-scoped directory:

```bash
CUDA_DEVICE=3 BENCHMARK_THREADS=4 bash benchmark/launch_full_benchmark.sh
```

After validating a run-scoped artifact set, publish it back into the canonical
`benchmark/results/` and `docs/assets/` paths with:

```bash
uv run python benchmark/publish_benchmark_artifacts.py \
	--results-json benchmark/results/runs/<run>/benchmark_results.json \
	--results-csv benchmark/results/runs/<run>/benchmark_results.csv \
	--scorecard-dir benchmark/results/runs/<run>/assets
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

## JAX-AMG exact and inverse comparisons

`jaxamgcg_f32` and `jaxamgcg_f64` select `JAXAMGCGSolver` only on NVIDIA
GPUs, for exact resistance and inverse landscape genetics. They never fall
back to CPU, approximate resistance, or another solver. Missing packages,
CUDA backends, or native shared libraries produce skipped records. See
[solver installation guidance](../docs/api/linear_solvers.md) for the optional
`jaxamg` extra and native AmgX prerequisites. The tested stack is JAX-AMG
0.1.4, AmgX 2.5.0, CUDA 12.9, JAX/JAXlib 0.10.0, and AMJax 0.0.3.

Timings retain one warm-up followed by three synchronized repeats. Tolerances
remain `rtol=atol=1e-3`; the existing exact grids (10, 100, and 1000 cells per
side), 20 sites, and inverse 100-by-100 case with LBFGS's 50-step budget remain
unchanged. No compilation, setup, or separate gradient timing is added.
AMJax builds its hierarchy outside timing and freezes it during inverse
optimization, while rebinding CG to each current matrix. JAX-AMG follows
upstream-managed native refresh, which may rebuild on each preconditioner
application inside the measured calls. Its refresh count is unknown, not zero
or one. No custom cache policy or setup-cost equivalence is implied.

Finish implementation, documentation, and correctness validation **before**
launching benchmarks, including smoke runs. To inspect the intended coverage
without executing workloads:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmark/jaxamg_comparison.py \
  --output-dir benchmark/results/runs/jaxamg-comparison --plan
```

Then run the same command without `--plan`. The directory must be new and
cannot be the canonical results directory. It includes exact JAXScape profiles,
AMJax CPU/GPU, Cholmod, PyAMG, Circuitscape, gdistance, and ResistanceGA where
installed; approximate and sensitivity profiles are excluded from this run.
Native CUDA/AmgX paths must already be configured as documented by upstream.
Use the Python environment containing that native installation.

The worker timeout is 180 seconds by default (including compilation, warm-up,
and all repeats); external tools default to 1800 seconds. These are watchdog
limits, not reported timing columns. Override with
`JAXSCAPE_BENCHMARK_WALLTIME_SECONDS` and
`JAXSCAPE_BENCHMARK_EXTERNAL_WALLTIME_SECONDS` and retain the manifest.
Failures and timeouts remain in the artifacts. JSON/CSV preserve individual
warm timings, inverse convergence/fit metrics, and JAX-AMG exact accuracy
against an untimed float64 Cholmod reference. Dependency versions, GPU kind,
visible-device selection, hierarchy behavior, invocation, and source commit
make the run traceable. Render scorecards with the existing renderer into the
same isolated directory:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmark/render_scorecard.py \
  --results-json benchmark/results/runs/jaxamg-comparison/benchmark_results.json \
  --output-dir benchmark/results/runs/jaxamg-comparison/assets
```

Keep these scoped results separate from the full published comparison. Do not
promote a partial run through `publish_benchmark_artifacts.py`; link its report
and raw artifacts from the benchmark documentation instead.

For a focused rerun after a correctness fix, add `--task inverse --jaxamg-only`
to select just the two JAX-AMG inverse profiles (or `--task resistance` for the
exact cases). Each invocation still writes a fresh manifest with its source
commit. Preserve the original run and identify any joined report records by
which run produced them; never silently replace the evidence of an earlier
failure.

The [14 October 2026 scoped report](../docs/benchmark_runs/jaxamg-2026-10-14/index.md)
contains the completed GPU 0 measurements, raw data, correction-run provenance,
and failed/unavailable cases. The existing full comparison assets were preserved.
