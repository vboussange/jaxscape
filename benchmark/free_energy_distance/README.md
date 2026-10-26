# Free-energy subset benchmark

This benchmark measures the production `FreeEnergyDistance` from issue
[#31](https://github.com/vboussange/jaxscape/issues/31), including its numerical
acceptance check and joint conductance/inverse-temperature derivatives.

## Reproduce

From the repository root, after installing JAXScape:

```bash
python benchmark/free_energy_distance/run.py \
  --output benchmark/free_energy_distance/results-cg.json
python benchmark/free_energy_distance/run.py \
  --sizes 8 --backends dense \
  --output benchmark/free_energy_distance/results-dense.json
python benchmark/free_energy_distance/run.py --numerical-only \
  --output benchmark/free_energy_distance/numerical.json
```

The recorded environment uses JAX 0.9.1, Equinox 0.13.2, and Lineax 0.1.0.
Float64 is enabled only in this benchmark process. GPU measurements are not
available on the measured machine. Raw results include package versions,
source revision and SHA256, device, thread environment, all timing samples,
compiler memory estimates, and nested JAX primitive/array-shape summaries.

## Workloads and measurement

A seed-20261026 raster contains independent uniform values in `[0.8, 1.4)`.
Rook-adjacent vertices have explicitly symmetric conductance `(x + y) / 2`.
The timed inverse temperature is `theta=1`. Three-column queries select three
nearby vertices; five-column queries have two sources and three targets.
Keeping the queried endpoints nearby lets larger sparse graphs exercise
scaling without making every distant logarithm unresolved.

Each objective is a signed weighted sum of the selected distances. Gradients
are taken jointly with respect to every raster cell and `theta`. The independent
float64 oracle constructs the reference transition matrix `P`, forms the
nonsymmetric killed walk `W`, and solves selected columns of `(I-W)^{-1}`.
It uses neither production operator assembly nor the custom VJP. Accuracy is
reported for distances, the scalar objective, raster gradients, and the
inverse-temperature gradient.

CG uses `max_steps=1000`, `rtol=1e-7, atol=1e-9` for float32 and
`rtol=1e-12, atol=1e-14` for float64. The optional dense comparison uses the
same mathematical model and query selections. This report makes no claim of
acceleration over the removed expected-cost RSP model, which computed a
different quantity.

Lowering/compilation is timed separately for setup, forward, value-and-gradient,
and backward. Setup is additionally timed as a warmed compiled operation.
Forward and value-and-gradient timings include graph/adjacency construction,
automatic solver state, selected solves, and acceptance checking, with inputs
already placed on the device. Seven warmed calls are synchronized with
`jax.block_until_ready` for each measurement. No other project test or benchmark
process runs concurrently during the recorded timing samples.

The check-overhead comparison compiles the same public workload with two
private validation helpers replaced only inside this benchmark process.
This removes the residual/rounding estimate and acceptance decision while
retaining the same operator, solver, selected columns, and tolerances. Outputs
are asserted bitwise equal. Checked and unchecked calls are interleaved in a
seeded random order. The resulting difference measures net compiled overhead,
including changes in fusion and error callbacks; it is not a public option to
bypass validation.

The numerical estimate uses one sparse residual block product, absolute row
sums times each solution column's maximum magnitude for the rounding allowance,
and reductions/logarithms. Its private distance allowance is
`16*sqrt(eps)*max(1, abs(distance))`. These measurements report observed errors
separately; passing the check does not certify floating-point or gradient
accuracy for arbitrary inputs.

## Results

Measured on 26 October 2026 with an Apple M1 Max CPU (`TFRT_CPU_0`),
JAX 0.9.1, Equinox 0.13.2, Lineax 0.1.0, and NumPy 2.5.4. The
implementation source was clean at `e62388d764c178baba5e48a488ee0eef0e738bb0`
(core implementation commit `ec6b413`). No thread-count environment overrides
were set. Raw results retain the source SHA256 and all seven samples.

All 12 CG cases and four small dense cases passed the production numerical
check. Times below are warmed medians in milliseconds. Check overhead is
relative to the same compiled solve with only residual/acceptance checking
removed by benchmark instrumentation.

| Vertices | Columns | Dtype | Forward | Value + gradient | Backward only | Check overhead |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 64 | 3 | float32 | 0.285 | 0.237 | 0.073 | 26.8% |
| 64 | 3 | float64 | 0.339 | 0.304 | 0.067 | 33.3% |
| 64 | 5 | float32 | 0.273 | 0.286 | 0.057 | 27.9% |
| 64 | 5 | float64 | 0.380 | 0.340 | 0.080 | 21.0% |
| 256 | 3 | float32 | 0.745 | 0.772 | 0.080 | 52.7% |
| 256 | 3 | float64 | 1.825 | 0.905 | 0.078 | 72.2% |
| 256 | 5 | float32 | 0.811 | 0.791 | 0.078 | 50.1% |
| 256 | 5 | float64 | 1.058 | 1.026 | 0.062 | 31.7% |
| 1024 | 3 | float32 | 2.813 | 2.860 | 0.109 | 60.8% |
| 1024 | 3 | float64 | 3.118 | 3.424 | 0.118 | 52.3% |
| 1024 | 5 | float32 | 3.094 | 3.226 | 0.142 | 56.2% |
| 1024 | 5 | float64 | 3.971 | 3.865 | 0.113 | 35.1% |

The check has a material constant-factor cost in these CG workloads
(21–72%). Dense 64-vertex cases had 0.7–5.6% measured overhead. Differences
between independently compiled forward and value-and-gradient programs are
affected by fusion, scheduling, and submillisecond timing variability; their
timings are not additive. The backward-only profile is measured separately.

| Dense vertices | Columns | Dtype | Forward (ms) | Value + gradient (ms) |
| ---: | ---: | --- | ---: | ---: |
| 64 | 3 | float32 | 0.176 | 0.217 |
| 64 | 3 | float64 | 0.139 | 0.198 |
| 64 | 5 | float32 | 0.176 | 0.185 |
| 64 | 5 | float64 | 0.157 | 0.198 |

Worst relative L2 errors across the timed cases, against independent float64
values/autodiff:

| Dtype | Distances | Raster gradient | Inverse-temperature gradient |
| --- | ---: | ---: | ---: |
| float32 | 7.81e-08 | 2.45e-07 | 1.26e-06 |
| float64 | 3.72e-14 | 3.50e-13 | 1.21e-12 |

Separately compiled warmed CG setup took
0.130–1.419 ms.
Dense setup returns `None`; its 0.003–0.004 ms measures dispatch of this
empty compiled operation. Compilation/lowering times for every case are in
`compile_ms` in the raw JSON; compilation is excluded from the tables above.
CG setup compilation ranged from 286.7 to 323.4 ms.
CG forward compilation ranged from 624.9 to 774.6 ms.
CG value and grad compilation ranged from 773.8 to 905.7 ms.
CG backward only compilation ranged from 96.9 to 109.6 ms.

## Backward work and memory

The backward-only benchmark first obtains a pullback through `jax.vjp` at the
public distance boundary. Its saved forward arrays become runtime arguments
to a compiled evaluation of that pullback's JAX program. This prevents
constant folding a fixed-input derivative into a precomputed gradient.
The result is checked against the gradient from the full value-and-gradient
computation.

Nested JAX primitive summaries and compiler memory estimates accompany every
case. For sparse subset queries, inspect `has_n_by_n_array` in both forward and
backward traces, and `solve_or_iteration_primitives` in the backward trace.
These are profiling evidence for the actual generated programs; ordinary unit
tests do not depend on private helper names or incidental solve ordering.

For every timed CG case, nested forward and backward traces contain no
`N × N` array. Every backward trace contains zero `custom_linear_solve`,
`lu`, `cholesky`, `triangular_solve`, `while`, or `pure_callback` primitives.
The saved-column contractions are present in the generated backward program.

For float64 with five selected columns:

| Vertices | Forward temporary (KiB) | Value + gradient temporary (KiB) | Backward temporary (KiB) | Backward saved arguments (KiB) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 126.0 | 138.0 | 11.8 | 24.8 |
| 256 | 498.4 | 543.2 | 47.2 | 96.1 |
| 1024 | 1987.9 | 2164.7 | 189.0 | 381.1 |

With `N` vertices, `M` stored sparse entries, and `K` selected inverse columns,
the saved forward solutions need `O(NK)` storage. The reused-column backward
computes `XG` in `O(NK^2)` and contracts the required sparse operator entries in
`O(MK)`. The numerical check adds `O(MK)` work without another solve. All-pairs
output still needs quadratic storage. Reported XLA temporary and argument
storage are compiler estimates, not measured peak process or device memory.

## Numerical range probes

`numerical.json` records the accepted and explicitly rejected sampled cases
on the same 8×8 raster with three selected vertices and the dense backend.
Accepted cases include independent distance and gradient comparisons; rejected
cases retain the standard error message. These samples describe this graph,
query, dtype, and backend. They do not define a universal admissible interval
for inverse temperature, and conservative rejection is intentional.

| Dtype | Accepted sampled theta | Rejected sampled theta |
| --- | --- | --- |
| float32 | 0.01, 0.1, 1 | 1e-12, 1e-08, 1e-05, 0.001, 5, 50, 500 |
| float64 | 0.001, 0.01, 0.1, 1, 5 | 1e-12, 1e-08, 1e-05, 50, 500 |

Across the accepted numerical probes, worst relative L2 errors were
1.17e-06 (distance), 8.39e-07 (raster gradient), and 6.42e-06 (theta gradient) for float32.
3.12e-14 (distance), 3.91e-14 (raster gradient), and 6.36e-13 (theta gradient) for float64.
All rejected samples raised the standard unresolved-logarithm error.
Small-theta cases expose ill-conditioning/cancellation; large-theta cases
expose tiny or underflowed queried entries. This finite sample supports the
private threshold calibration and separate gradient checks, without asserting
a floating-point accuracy certificate or a universal usable theta range.
