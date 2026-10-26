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
| Free-energy distance | `JAXScape / CG / theta=0.01,0.1,1 / f64 (CPU/GPU)` | custom cases via the shared `build_case` helper | 10×10 and 100×100 grids, 20 selected sites; independent value and directional-gradient checks accompany timings. |
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

## Free-energy profiles

```bash
uv run --extra benchmark python benchmark/jaxscape/free_energy_distance.py
```

This uses the existing standalone runner and writes the same JSON/CSV record
format under `benchmark/results/tasks/free_energy_distance/`. Full-suite and
smoke runs also include FED. Cases reuse the resistance raster generator, seeds,
interior-site selection and `cost_conductance`; a worker receives the realized
case file so reference and production calculations use identical arrays.

The explicit float64 profile is `lineax.CG(rtol=1e-10, atol=1e-90,
max_steps=10000, stabilise_every=None)`. The small absolute tolerance resolves
distant entries, while disabling periodic residual replacement avoids injecting
the near-source rounding floor. The independent production guard remains
essential because recursive residuals may drift. These are measured workload
settings, not universal defaults or an automatic dtype promotion.

`median_seconds` is warmed forward time including graph construction and solver
state rebuilding. `metrics` additionally contains lowering/compilation time,
separate setup and value-and-gradient samples, device placement, versions,
precision, solver settings and independent accuracy checks. Timings use the
shared synchronization helper, exclude compilation and inputs' device transfer,
and include setup on every changed-landscape call. Setup is also measured
separately; do not add it again to forward time. Errors produce `failed` records
with the available metadata, rather than a favorable timing without validation.

The independent SciPy reference constructs the nonsymmetric killed walk
`W = P * exp(-theta * cost)` from NumPy grid edges and solves selected columns of
`I-W`. All 380 off-diagonal values must meet
`max(abs(error)/max(1,abs(reference))) < 1e-7`. A signed weighted objective checks
three deterministic landscape directions and theta against central differences
with scaled discrepancy below `1e-6`. Steps are `1e-4` for normalized landscape
directions and `theta*1e-4` for theta. These allow finite-difference truncation and
cancellation error; they do not certify every gradient component.

For bounded larger cases or an ordinary-tolerance negative control, use
`build_case(..., size=200, seed=1)` and `run_profile(case, theta=0.1,
device=jax.devices()[0], repeats=3)`. The optional `solver_settings` argument can
record rejected ordinary-CG runs. No million-cell or arbitrary-temperature
support is promised. Existing subprocess walltime limits and failure records
apply to normal runs; the direct profiling helper leaves resource limits to its
caller.

The optional ConScape.jl comparison is deferred: the existing Julia environment
and adapters contain Circuitscape resistance calculations only. A matching FED
comparison needs a new package setup and adapter that constructs the same
reference walk, reciprocal costs and target normalization, validates values,
and times construction/factorization rather than a cached accessor. That is
additional integration beyond this contained revision. SciPy is an independent
correctness reference here, not a claim about ConScape performance.

### Validated revision profile (26 October 2026)

[Raw records](jaxscape/free_energy_validation.json) retain inputs' and production
source hashes, software versions, compilation/setup times, all timing samples,
device placement, directional checks and expected numerical failures. These
measurements use the shared runner above on an Apple M1 Max CPU (JAX 0.9.1,
Lineax 0.1.0) and an RTX 3090 (JAX 0.10.0, Lineax 0.1.0). The original prototype
timings are not reused. Each device's reference uses its exact realized array.

All rows use 20 sites and float64. Times are warmed medians of three synchronized
samples, except 500×500 (one sample after warming); compilation is excluded.
Value errors cover all 380 off-diagonal entries. Gradient errors are the largest
scaled discrepancy across three signed-objective landscape directions and theta.

| Device | Grid | Theta | Forward (ms) | Value + gradient (ms) | Max value error | Max gradient discrepancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 10×10 | 0.1 | 7.7 | 7.8 | 1.87e-15 | 4.91e-09 |
| CPU | 100×100 | 0.01 | 5603.2 | 5303.4 | 1.86e-15 | 8.25e-09 |
| CPU | 100×100 | 0.1 | 2037.9 | 2016.4 | 5.69e-16 | 6.87e-09 |
| CPU | 100×100 | 1 | 732.9 | 745.3 | 4.27e-15 | 2.97e-09 |
| GPU | 10×10 | 0.1 | 16.4 | 16.6 | 1.36e-15 | 4.91e-09 |
| GPU | 100×100 | 0.01 | 450.8 | 453.0 | 1.86e-15 | 8.23e-09 |
| GPU | 100×100 | 0.1 | 180.5 | 181.9 | 4.32e-16 | 6.87e-09 |
| GPU | 100×100 | 1 | 71.1 | 80.3 | 3.24e-15 | 2.97e-09 |
| GPU | 200×200 | 0.1 | 539.0 | 538.1 | 5.36e-16 | 9.05e-09 |
| GPU | 500×500 | 0.01 | 8536.5 | 8564.7 | 1.03e-15 | 1.85e-08 |

CPU negative controls at 100×100, theta=0.1 and 1, with `atol=1e-14` and
`stabilise_every=10` still raised the existing unresolved-logarithm error.
Float32 values/gradients and all optional solvers passed the public regression
suite (105 tests in total); this large-grid GPU profile was measured only in
float64. The 200/500 cases are bounded examples, not general scaling guarantees.

At 100×100/theta=1, seven randomized paired measurements found guard overhead
of 2.4% on CPU and 0.9% on GPU relative to the same compiled solver
work with only guard routines bypassed in temporary measurement instrumentation.
This is a net compiler-dependent cost, not a public switch or a CI threshold.
GPU recompilation changes values by at most 1.32e-14 scaled error; no validation
result above bypasses the guard. Nested traces contain no N×N arrays in either
direction and no solve/iteration/callback primitives in the backward pass.
The numerical solve runs in JAX on the GPU; the retained forward host callbacks
only report errors. The guard uses four sparse block products and no extra solve.

The revised guard reuses the existing SPD assembly, selected solves, final
logarithm acceptance check and distance-owned VJP. The benchmark reuses case
generation, site selection, conductances, synchronized timing, worker execution
and result records; its small independent SciPy oracle is needed to check values
and finite differences without reusing the production algebra.
