# Benchmarking

JAXScape includes a lightweight benchmark harness at
`benchmark/benchmark_distances.py`. The script focuses on the three workloads
tracked in the benchmarking issue:

- **Connectivity analysis**: forward-pass connectivity on a raster graph
- **Sensitivity analysis**: gradient of connectivity with respect to permeability
- **Inverse landscape genetics**: resistance-distance fitting with `Optimistix`

All runs are written to JSON so they can be compared across machines, devices,
and external tools.

## Install benchmark dependencies

The connectivity and sensitivity workloads only require the base package:

```bash
python -m pip install -e .
```

The inverse-landscape-genetics benchmark additionally needs `optimistix`:

```bash
python -m pip install -e '.[examples]'
```

## Benchmark synthetic landscapes

Run all benchmark scenarios on CPU:

```bash
python benchmark/benchmark_distances.py \
  --scenario all \
  --device cpu \
  --distance resistance \
  --grid-sizes 8 12 16 \
  --repeats 3
```

If a GPU-enabled JAX installation is available, benchmark both CPU and GPU:

```bash
python benchmark/benchmark_distances.py \
  --scenario all \
  --device cpu gpu \
  --distance resistance \
  --grid-sizes 8 12 16 \
  --repeats 3
```

## Benchmark a shared raster

To compare JAXScape against an external benchmark suite such as
[`BigTests`](https://github.com/Circuitscape/BigTests/), first convert the test
landscape to `.npy` or comma-delimited `.csv`, then point the harness at that
raster:

```bash
python benchmark/benchmark_distances.py \
  --scenario connectivity \
  --device cpu gpu \
  --distance resistance \
  --landscape-path /path/to/bigtests_landscape.npy \
  --repeats 5
```

The resulting JSON file can be committed separately, plotted in a notebook, or
combined with timings collected from `Circuitscape.jl`, `conefor`, `samc`, or
other reference tools.

## Output format

Each run records:

- the scenario name
- the device used
- the distance metric
- the raster shape
- all measured runtimes
- the median runtime

Additional scenario-specific metadata is included as well, such as the
connectivity value, gradient norm, or inverse-optimization status.
