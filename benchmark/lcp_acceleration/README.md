# Least-cost-path before/after benchmark

`run.py` compares the current default `LCPDistance()` with its prior
implementation loaded by `git show` from the explicit `--baseline-ref` (default
`33e609b`). Both use the public API and current graph construction/dependencies.
The frozen module exists only in the benchmark process; there is no maintained
legacy backend in the library. See [the distance API](../../docs/api/distances.md)
for the intentional change in tie gradients.

## Reproduce

From the repository root, with test dependencies installed:

```bash
uv run pytest tests
uv run python -m benchmark.lcp_acceleration.run --output benchmark/lcp_acceleration/production_cpu.json --device cpu
```

The default suite uses seed 42, float32, two warmups and seven samples per
executable: 32×32 and 64×64 random grids with one/eight sources; a 64×64 clipped
ridge; a uniform 32×32 grid with exact ties; and a 1×512 corridor requiring 511
hops. Grids use 19 sampled targets; the corridor uses its endpoints. Positive
raster values are cell costs and rook-edge conductance is `2/(cost_u+cost_v)`.
The differentiated loss sums selected distances with respect to the raster.

Optional `--device gpu` and `--dtype float64` runs write separate reports. An
unavailable requested GPU fails clearly. For a quick run use `--sizes 8
--sources 1 2 --corridor-size 32 --repeats 3`. The baseline revision must be
present locally; fetch it first when using a shallow clone.

## Measurement contract

Lowering/compilation are timed separately; the persistent compilation cache is
disabled. Input creation/placement, warmups and correctness checks are outside
the timings. Graph/adjacency construction are inside. Every sample synchronizes
the complete output with `jax.block_until_ready`; method/task order is randomized
per repetition. All raw samples, medians, sample order, versions, hardware,
device availability, exact revisions, source hashes and input hashes are saved.
`completed_at_utc` distinguishes complete reports from interrupted partial data.

SciPy Dijkstra independently checks values using explicitly constructed
reciprocal costs; these workloads have no positive duplicate coordinates.
Differentiated scalar values are checked too and gradients must be finite.
Gradient differences from the old backend are descriptive, because ties can
legitimately differ. Mathematical VJP correctness belongs to the public API
tests; ordinary tests contain no timing-ratio assertions.

XLA temporary/argument/output/alias byte counts are **compiler estimates, not
measured peak process or device memory**. CPU results do not establish GPU
performance, large all-pairs scalability or universal speedups. The corridor
has no early-stopping advantage. Compare software versions and hardware load
before interpreting compilation or execution differences between runs.

## Production measurements (23 October 2026)

[Raw production report](production_cpu.json): Apple M1 Max CPU, float32,
JAX/jaxlib 0.9.1, Equinox 0.13.2, NumPy 2.4.2, SciPy 1.16.2.
Clean production revision `d88ad551fb48a70f90047b2ec71b26fbfefc289b`;
baseline module `33e609bc846e581273777573a269f59d504c0286`.
GPU was unavailable. All values passed the independent SciPy checks and all
gradients were finite. Medians below exclude compilation, which is separately
reported alongside every raw sample in the JSON.

| Workload | Sources | Before forward (ms) | After forward (ms) | Before value + grad (ms) | After value + grad (ms) | Value + grad speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 random | 1 | 4.299 | 0.467 | 40.342 | 1.219 | 33.1× |
| 32×32 random | 8 | 18.602 | 1.496 | 198.861 | 2.984 | 66.6× |
| 64×64 random | 1 | 67.920 | 1.895 | 657.862 | 7.845 | 83.9× |
| 64×64 random | 8 | 283.529 | 10.708 | 2309.346 | 21.388 | 108.0× |
| 64×64 clipped | 1 | 68.183 | 1.998 | 674.227 | 7.927 | 85.1× |
| 32×32 uniform | 1 | 4.225 | 0.464 | 39.704 | 1.149 | 34.5× |
| 1×512 corridor | 1 | 2.785 | 2.866 | 14.740 | 7.215 | 2.0× |

For 64×64 with eight sources, XLA estimated value-plus-gradient temporary
storage decreased from 515.8 MiB to 2.34 MiB. This is not a measured memory peak.
The corridor forward pass was 2.9% slower: it needs the full relaxation bound,
and the new implementation also validates the final tight graph. Its complete
value-plus-gradient evaluation was still 2.0× faster.

Against the original same-hardware/same-JAX random-grid prototype medians,
production value-plus-gradient times ranged from 1.2% faster to 8.6% slower;
the clipped case was 4.6% slower. These separate runs show no material
regression against the prototype. Validation and stable reciprocal derivative
arithmetic add work, but these measurements do not isolate their overhead
from run-to-run noise. Gradient relative L2 differences from the historical
backend were at most 2.2e-7 in these sampled cases; the analytical tie
counterexample in the API guide demonstrates that agreement is not universal.

## Historical investigation (22 October 2026)

Retained original raw evidence:

- [DAG random-grid timings and correctness](historical/dag_results.json)
- [Clipped-ridge timings and correctness](historical/published_formula.json)
- [Checksums](historical/SHA256SUMS)

Those runs used Apple M1 Max CPU, JAX 0.9.1, Equinox 0.13.2, float32 and revision
`5455ca4f0a5ebfe5f27f0b24c672e8e6948c73b4`. The unreleased prototype measured
approximately 35–105× random-grid value-plus-gradient speedups and 88× for the
clipped ridge. The original JSON retains every raw sample, compile time and
reported correctness metric. Dead prototype solver variants are not maintained
here; the public harness replaces the investigation harness for reproducible
production measurements. Tree or active-scan variants in historical raw data
are not production backend choices.
