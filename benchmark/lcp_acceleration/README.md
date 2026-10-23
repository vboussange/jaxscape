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

## Production measurements

Production results are recorded separately in `production_cpu.json` once the
correctness checks pass. The original prototype results below are historical
evidence, not measurements of the production implementation.

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
