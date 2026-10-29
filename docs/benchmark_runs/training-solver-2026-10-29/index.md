# Training-time resistance solver: 29 October 2026

This run chooses the solver for exact resistance distances inside compiled
inverse landscape genetics training. It was prompted by
[vboussange/ilg-toolkit#117](https://github.com/vboussange/ilg-toolkit/issues/117).
It compares three GPU directions on the same problem:

- rebuilding the AMJax algebraic multigrid (AMG) hierarchy from the current
  conductance surface;
- a sparse direct solver, cuDSS, through [spineax](https://github.com/johnviljoen/spineax)
  ([#1](https://github.com/vboussange/jaxscape/issues/1));
- JAX-AMG, the AmgX bindings whose integration was reverted
  ([#19](https://github.com/vboussange/jaxscape/issues/19)–[#22](https://github.com/vboussange/jaxscape/issues/22), [#23](https://github.com/vboussange/jaxscape/pull/23), [#29](https://github.com/vboussange/jaxscape/pull/29)).

**Decision: add a cuDSS direct solver through spineax.** On the same
surfaces, and with its factorization state passed from one step to the
next, it solves 20 right-hand sides 6–11× faster than a perfectly rebuilt AMG
preconditioner at 1000×1000. It also skips that preconditioner's 3 s host
rebuild. It agrees with CHOLMOD to within 3·10⁻⁹ (relative), and its cost does
not depend on conductance contrast. AMG rebuilding is kept as a future option for
grids too large to factorize. JAX-AMG is deferred: used correctly it works,
but it is slower than cuDSS, solves right-hand sides one at a time and needs a
native AmgX build.

## Problem

ilg-toolkit learns a conductance surface. Every training step computes
effective resistances among the sampled vertices of each region and species,
inside `jax.jit`, and differentiates them. The graph is JAXScape's
four-neighbour `GridGraph`: edge conductance is the mean of its endpoints, the
last vertex is grounded, and the arithmetic is float64. Since 0.0.11,
JAXScape's exact-resistance gradient reuses the forward solutions, so a
training step costs one multi-RHS forward solve and no backward solve.

ilg-toolkit currently uses Lineax CG (`rtol=atol=1e-6`, `max_steps=1000`),
optionally preconditioned by an `AMJaxCGSolver` hierarchy built once from the
uniform graph. As the surface becomes heterogeneous, CG exhausts its budget
and training aborts. The fix must keep the computation on the GPU, with no
host round trip per step.

## Method

- **Hardware.** NVIDIA GeForce RTX 3090 (24 GiB), driver 580.178.04, one GPU
  per experiment. The host is an AMD EPYC 7742 shared with other jobs, so host
  timings are noisy.
- **Problem.** The grounded Laplacian described above, with 20 terminal
  vertices and CG at `rtol=atol=1e-6`. Times are for jitted calls, synchronized
  with `block_until_ready`. Unless noted, each is the median of three warm calls.
- **Surfaces** ([`amg/common.py`](amg/common.py)):
    - `uniform`.
    - `grfK`: log₁₀ C is a smooth Gaussian random field with a correlation
      length of 7.5 % of the side, scaled to C ∈ [10⁻ᴷ, 1].
    - `barrier`: C = 1 with three one-percent-wide barriers at 10⁻⁴.
    - `floor`: a 3-decade field with its lowest-quartile area clamped to 10⁻⁶,
      mimicking a softplus output driven to a 10⁻⁶ floor.
- **Software.**
    - AMG and JAX-AMG experiments: JAXScape source at
      [`005b92b`](https://github.com/vboussange/jaxscape/commit/005b92b) (0.0.12)
      with jax 0.10.0 (CUDA 12), lineax 0.1.0, equinox 0.13.6, amjax 0.0.3,
      pyamg 5.3.0 and cholespy 2.2.0.
    - JAX-AMG additionally: jaxamg 0.1.4 and unmodified AmgX 2.5.0 (revision
      `cc1cebdb`), built against CUDA 12.9.
    - cuDSS experiment: a separate environment with jax 0.10.0 (CUDA 12),
      lineax 0.1.1, equinox 0.13.8, spineax 0.0.5 built from source at
      `97dae03f` and nvidia-cudss-cu12 0.8.0.10 (see [build notes](cudss/build.md)).
- **Exceptions.**
    - The cuDSS phase benchmark ([`bench_spineax.py`](cudss/bench_spineax.py),
      [`grad_check.py`](cudss/grad_check.py)) uses an uncorrelated log-normal
      field over [10⁻², 10²] and grounds vertex 0.
    - [`contrast_check.py`](cudss/contrast_check.py) repeats the cuDSS accuracy
      and timing on the AMG experiment's surfaces and terminals.
    - The JAX-AMG benchmark uses a smooth field over [10⁻², 10²].
- **References.** Every accuracy figure compares effective resistances, pair
  by pair, with CHOLMOD (cholespy) on the same matrix. The metric is the
  maximum relative error over off-diagonal pairs.

## 1. AMG: frozen versus rebuilt hierarchies

Scripts are in [`amg/`](amg/run.py); the tables come from
[`summary_tables.txt`](amg/results/summary_tables.txt).

| Variant | Preconditioner |
| --- | --- |
| plain / Jacobi | Unpreconditioned CG, or diagonal (Jacobi) preconditioning from the current matrix |
| B | AMJax hierarchy frozen at the uniform graph: PyAMG smoothed aggregation, `pinv` coarse solve. This is ilg-toolkit's current setup. |
| B_fine | B with only the finest level's matrix and Jacobi diagonal replaced by the current ones |
| C | PyAMG smoothed aggregation rebuilt from the current matrix |
| C θ=0.25 | C with a coefficient-aware strength of connection |
| D | C, reusing B's aggregates |
| E | Plain (unsmoothed) aggregation on B's aggregates, recomputed from the current matrix |

**CG iterations** for the worst of the 20 right-hand sides, at 100² / 300² / 1000².
`>N` means the cap was reached. C and D give the same counts except on the
300² `floor` surface (40 against 41).

| Surface | Plain CG | Jacobi | B (today) | B_fine | C = D | C θ=0.25 | E |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 500 / 1429 / – | 503 / 1451 / – | 24 / 25 / 21 | 24 / 25 / 21 | 24 / 25 / 21 | 82 / 253 / – | 50 / 86 / 158 |
| grf2 | 1522 / 4962 / – | 553 / 1651 / – | 145 / 118 / 145 | 57 / 75 / 90 | 22 / 25 / 22 | 164 / 530 / – | 56 / 97 / 175 |
| grf4 | 6204 / >20000 / – | 598 / 1824 / – | **1159 / 1057 / 1164** | 123 / 188 / 310 | 24 / 29 / 24 | 178 / 570 / – | 65 / 119 / 239 |
| grf6 | >20000 / >5000 / – | 626 / 1922 / – | 9945 / >5000 / >1200 | 250 / 436 / 686 | 29 / 39 / 29 | 194 / 594 / – | 76 / 141 / 286 |
| barrier | 601 / 2797 / – | 521 / 2031 / – | 67 / 309 / 393 | 25 / 73 / 82 | 26 / 46 / 40 | 86 / 353 / – | 53 / 180 / 332 |
| floor | >20000 / >5000 / – | 624 / 1998 / – | 3385 / 4918 / >1200 | 128 / 300 / 677 | 30 / 40 / 31 | 188 / 568 / – | 63 / 115 / 255 |

**Warm solve time for 20 RHS at 1000², in seconds.** Capped runs are lower bounds.

| Surface | B (today) | B_fine | C | E |
| --- | ---: | ---: | ---: | ---: |
| uniform | 0.651 | 0.650 | 0.649 | 4.31 |
| grf4 | 34.8 | 9.31 | 0.744 | 6.55 |
| grf6 | ≥35.9 | 20.7 | 0.891 | 7.83 |
| barrier | 11.7 | 2.47 | 1.22 | 9.08 |
| floor | ≥35.9 | 20.3 | 0.953 | 6.99 |

**Rebuild cost (C and D), as total host time with the PyAMG share in brackets:**

- 100²: 0.07 s [0.02 s], against a 0.011–0.014 s solve.
- 300²: 0.31–0.50 s [0.24–0.42 s], against a 0.07–0.13 s solve.
- 1000²: 2.9–3.8 s [2.7–3.6 s], against a 0.65–1.22 s solve. Two outliers,
  7.7 s and 4.6 s, coincided with other jobs on the host.
- The first conversion in a process costs 8–18 s, because AMJax's eager JAX
  operations compile.
- E recomputed entirely in JAX ([`ejax.py`](amg/ejax.py)) takes 1.2 ms at 300²
  and 2.0 ms at 1000². It matches PyAMG to within 4·10⁻¹⁴, with fixed shapes.

**Accuracy.** C and D are within 7.2·10⁻⁷ everywhere, and B within 2.7·10⁻⁵
when it converges. A capped solve returns silently wrong values under
`throw=False`: plain CG reaches a relative error of 2.2 on the 100² `floor`
surface, and B reaches 3.7·10⁻³ on the 1000² `grf6` surface.

**Recompilation** ([`recompile_test.py`](amg/recompile_test.py),
[`recompile_100.json`](amg/results/recompile_100.json)):

- Passing a rebuilt `AMJaxCGSolverState` into an `eqx.filter_jit`-compiled
  objective retraces and recompiles every time, even for a bit-identical
  rebuild: 2.7–3.4 s at 100², both forward and for the gradient. The state
  holds a closure-converted `lx.FunctionLinearOperator` whose jaxpr is
  compared by identity.
- Passing the raw `amjax.MultilevelSolver` pytree, and building the
  preconditioner inside the traced function, does not retrace.
- B_fine, C and D had B's tree structure and leaf shapes in all 18
  size/surface cases. This is observed, not guaranteed: SciPy drops exact
  zeros from sparse products.

**Findings.**

1. **The frozen hierarchy causes the failure.** Its iteration count depends on
   contrast, not grid size: about 22, 130, 1100 and 10⁴ for 0, 2, 4 and 6
   decades. It crosses ilg-toolkit's 1000-step budget at 4 decades, at every
   size.
2. **Rebuilding fixes the iteration count** (21–46 everywhere), and makes CG
   10–47× faster than B at 1000² on heterogeneous surfaces.
3. **C and D are the same.** PyAMG's default strength of connection makes
   aggregation depend only on the sparsity pattern: C's aggregates equalled
   B's in every case. The cost lies in prolongator smoothing and the Galerkin
   products, not in aggregation.
4. **The cheap GPU-only variants fall short.** B_fine and E help, but still
   need hundreds of iterations at 1000². A smoothed-aggregation refresh in JAX
   with a fixed sparsity pattern should reach C's quality at millisecond cost,
   but it was not built.

## 2. cuDSS through spineax

Scripts are in [`cudss/`](cudss/bench_spineax.py). spineax exposes cuDSS's
analysis, factorization, refactorization and solve phases as XLA FFI calls.
Each factorization is tracked by a token whose `id` is used up by every
numeric phase.

**Phases** ([`bench_small.json`](cudss/results/bench_small.json),
[`bench_1000.json`](cudss/results/bench_1000.json),
[`grad_check.log`](cudss/results/grad_check.log),
[`grad_check_1000.log`](cudss/results/grad_check_1000.log)).
Float64, conductance over [10⁻², 10²]. A "step" is a jitted `value_and_grad`
of the summed pairwise effective resistances, using spineax's raw API.

| Grid | Analysis | Refactorize | Solve, 20 / 100 RHS | Step, token passed on, 20 / 100 terminals | Step, token reused | Error vs CHOLMOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100² | 54 ms | 1.3 ms | 0.9 / 1.9 ms | 2.8 / 5.0 ms | 55 ms | 7.6·10⁻¹³ |
| 300² | 0.37 s | 6.1 ms | 4.3 / 16 ms | 15 / 40 ms | 0.38 s | 6.9·10⁻¹² |
| 1000² | 4.7 s | 70 ms | 44 / 163 ms | 152 / 433 ms | 4.85 s | 8.8·10⁻¹¹ |

For comparison, CPU CHOLMOD at 1000² takes 15.3 s to analyse and factorize,
then 0.73 s for 20 RHS or 3.6 s for 100.

**Same surfaces as the AMG experiment** ([`contrast_check.log`](cudss/results/contrast_check.log)).
These times cover refactorizing and solving 20 RHS, with the token passed on
between calls.

| Grid | Time, every surface | Worst error vs CHOLMOD | Worst relative residual |
| --- | ---: | --- | --- |
| 100² | 1.8–1.9 ms | 2.3·10⁻⁹ (`floor`) | 1.9·10⁻⁸ |
| 300² | 10.5–10.9 ms | 2.1·10⁻¹⁰ (`floor`) | 5.0·10⁻⁸ |
| 1000² | 106–113 ms | 5.7·10⁻¹⁰ (`barrier`) | 7.8·10⁻¹¹ |

Cost does not depend on contrast. With the token passed on, no call
re-analysed.

**Behaviour checked:**

- **No recompilation or re-analysis.** With the token passed on, re-running
  the compiled step on new values left the jit cache at size 1 and spineax's
  `rebuild_count()` at 0.
- **A reused token re-analyses.** Passing a used-up token makes spineax
  rebuild the factorization from scratch: 3 of 4 steps in each check. This
  costs the analysis time at every step (4.85 s at 1000²).
- **Gradients are correct.** They agree with central finite differences to
  within 1.3·10⁻⁸ to 4.5·10⁻⁶ (relative), from 100² to 1000² with 20 or 100
  terminals.
- **The raw solve batches over vmap.** Vmapping the raw solve over
  right-hand sides runs one multi-RHS cuDSS solve, and its gradient matches
  the unbatched one to within 10⁻¹².
- **spineax's own Lineax solver has a bug.** Its `CuDSS` class fails when
  `jax.vmap(lx.linear_solve)` is differentiated ("Expected cotangent type
  float64[1,nnz] … got float64[k,nnz]"). This is the pattern JAXScape's
  batched solve uses.
- **Installation.** The PyPI wheel is CUDA 13 only. The CUDA 12 source build
  took about 45 s, needed no `nvcc`, and passed spineax's 63 tests.

## 3. JAX-AMG 0.1.4 and AmgX 2.5.0

Scripts and logs are in [`jaxamg/`](jaxamg/scripts/bench_jaxamg.py). The
surface is a smooth 4-decade field.

**Why the reverted integration was 100–300× slower than AMJax:**

- **Setup runs on every call.** Every preconditioner application, so every CG
  iteration, called `AMGX_matrix_replace_coefficients` and then a full
  `AMGX_solver_resetup`. At 300² that costs 15.8 ms per application, against
  1.4 ms without the resetup and 0.23 ms for an AMJax V-cycle.
- **Every call synchronises.** It syncs the stream and device and copies the
  sparsity pattern to the host, where it is hashed to build the cache key.
- **Right-hand sides run one at a time** (`vmap_method="sequential"`).
- **The cache holds a single entry.**

| | 300² | 1000² |
| --- | ---: | ---: |
| AMJax frozen at uniform: iterations / 20-RHS solve | ~700 / 2.06 s | 673 / – |
| AMJax rebuilt: iterations / 20-RHS solve | 24 / 0.058 s | 25 / 0.77 s |
| JAX-AMG V-cycle inside Lineax CG, default refresh: 20 RHS | 4.66 s | – |
| JAX-AMG V-cycle inside Lineax CG, setup reused: 20 RHS | 0.38 s | – |
| AmgX PCG: iterations | 15 | 18 |
| AmgX PCG per RHS, resetup / setup reused | 24 / 9.9 ms | 77 / 39 ms |
| AmgX PCG value and gradient, 20 RHS, resetup every solve / setup once | 0.95 / 0.41 s | 3.0 / 1.58 s |

A hierarchy kept from another matrix is unsafe
([`check_reuse_semantics.log`](jaxamg/logs/check_reuse_semantics.log)).
Set up on the uniform matrix and reused on the heterogeneous one, it made
Lineax CG stop after 8 steps and report success, with a true relative residual
of 3.8. Keeping coarsening but recomputing values (`structure_reuse_levels`)
costs 5.2 ms per application and needs 20 iterations instead of 13.

## 4. Comparison

Forward solve for 20 RHS on a 4-decade surface. The JAX-AMG field differs
(see Method).

| Direction | 300² | 1000² | Iterations | Error at 1000² |
| --- | --- | --- | --- | --- |
| AMJax frozen (today) | 3.0 s | 34.8 s | ~1100: over budget | 1.7·10⁻⁷ |
| AMJax rebuilt every step | 0.084 s + 0.3–0.5 s host | 0.74 s + 2.9–3.8 s host | 24–29 | 1.5·10⁻⁷ |
| AmgX PCG via JAX-AMG, setup once per step | 0.18 s | ≈0.8 s (39 ms × 20, extrapolated) | 15–18 | 5.1·10⁻⁷ relative residual |
| **cuDSS via spineax** | **0.011 s** | **0.11 s** | direct | **5.7·10⁻¹²** |

## Decision and consequences

- **cuDSS through spineax becomes JAXScape's GPU direct solver**, as an
  optional extra. The reasons:
    - Iteration-budget failures disappear, and there is no hierarchy to go
      stale.
    - The analysis depends only on the sparsity pattern, so one state serves
      every conductance surface on the same grid.
    - It is the fastest option measured, and exact.
- **The design has to respect three constraints:**
    - Distance calls must return the refreshed solver state, so a training
      loop can pass it from one step to the next.
    - The integration must call spineax's raw phases, with JAXScape's own
      resistance gradients, rather than spineax's Lineax solver.
    - spineax's cuDSS 1.0 support (upstream issues #29 and #30) and its
      CUDA 13-only wheel affect packaging.
- **AMJax keeps its current, frozen behaviour.** If a grid becomes too large to
  factorize, the preferred AMG path is a smoothed-aggregation refresh in JAX
  over the fixed aggregates, with the raw hierarchy passed as the state. It is
  preferred to host rebuilds.
- **JAX-AMG ([#19](https://github.com/vboussange/jaxscape/issues/19)–[#22](https://github.com/vboussange/jaxscape/issues/22)) is deferred.**
  It could be revisited if JAX-AMG gains multi-RHS solves and wheels, or as
  AmgX PCG with one setup per step.

## Follow-up

- [#41](https://github.com/vboussange/jaxscape/issues/41) specifies `CuDSSSolver` and the state-returning `ResistanceDistance` queries. It supersedes [#1](https://github.com/vboussange/jaxscape/issues/1).
- [vboussange/ilg-toolkit#141](https://github.com/vboussange/ilg-toolkit/issues/141), blocked by #41, adopts the solver in training. It also adds a failure policy that names the region, species and step.
- [#19](https://github.com/vboussange/jaxscape/issues/19)–[#22](https://github.com/vboussange/jaxscape/issues/22) are closed as not planned. The reopening conditions are on [#19](https://github.com/vboussange/jaxscape/issues/19#issuecomment-6491249959).

## Limitations

- **One GPU model.** All runs used a consumer RTX 3090, which has weak FP64
  throughput. Data-center GPUs will shift the absolute numbers.
- **cuDSS was measured through spineax directly.** A JAXScape solver adds
  BCOO-to-CSR value mapping and state handling, which were not measured.
- **Synthetic surfaces only.** Iteration counts are the worst of 20 RHS on
  synthetic surfaces. No training trajectory was replayed, and how fast a
  surface drifts between steps was not measured.
- **Reduced runs.** Some AMG cells used lower caps or a single warm call:
    - 300² `grf6`, `barrier` and `floor`: cap 5000.
    - 1000²: cap 1200.
    - 1000² plain CG, Jacobi CG and θ=0.25 were not run.
- **JAX-AMG versions.** JAX-AMG 0.1.5 and upstream `main` were read but not
  benchmarked.
- **Memory.** No memory was measured, including cuDSS factor memory beyond
  1000².

## Reproduce

Each directory has an `env.sh` recording its environment. `PY` is that
environment's Python.

```bash
# AMG (from amg/; GPU 1)
source env.sh
$PY run.py --size 100
$PY run.py --size 300 --surfaces uniform grf2 grf4
bash chain.sh
$PY summarize.py
# cuDSS (from cudss/; GPU 2)
source env.sh
$PY bench_spineax.py --sizes 100 300 --out results/bench_small.json
$PY bench_spineax.py --sizes 1000 --out results/bench_1000.json
$PY grad_check.py --sizes 100 300
$PY grad_check.py --sizes 1000 --k 20 100
$PY contrast_check.py --sizes 100 300 1000
# JAX-AMG (from jaxamg/scripts/; GPU 3; needs AMGX_BUILD and CUDA_HOME)
source env.sh
$PY bench_jaxamg.py 300
$PY bench_scaling.py 300
$PY bench_scaling.py 1000
$PY bench_structure_reuse.py
$PY check_reuse_semantics.py
```

The scripts and logs are kept as run, except for these changes:

- the `env.sh` files read machine-specific paths from variables;
- the cuDSS environment file is new;
- `contrast_check.py` was tidied after its run, with no change to what it
  computes.
