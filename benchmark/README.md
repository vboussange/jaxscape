# Benchmark

The benchmark compares JAXScape with the software that ecologists use for
connectivity analysis, on three questions: E1, isolation between populations;
E2, habitat connectivity; E3, corridors and pinch points. Its results make the
[Benchmark page](https://vboussange.github.io/jaxscape/benchmark/) of the docs.

This directory is a uv project of its own, outside the root project, pinned to a
JAXScape release from PyPI (`jaxscape==0.0.16` in `pyproject.toml`). Run every
command from this directory:

```bash
uv sync
./install.sh                                  # the pinned toolchains and the Maryland raster
uv run python run.py results/smoke --smoke   # every cell once, on 32 x 32 rasters, small windows
nohup uv run python run.py results/v0.0.16 > run.log 2>&1 &   # the full run
uv run python report.py results/v0.0.16      # the CSV, the figures and the page
uv run pytest                                 # tests of the runner and the report
```

Smoke mode takes a new run directory (remove `results/smoke` to run it again),
and exits with an error when a cell fails or lacks its agreement. A full run
appends one record per cell to `records.jsonl` as each cell finishes; when it
restarts, it skips the cells recorded there. To benchmark a later release,
change the pin, run, and report.

## Layout

- `config.toml`: how cells run (timeout, repeats, cores, GPU), the landscape,
  the raster sides and shared parameters of each distance, the windows of E2
  and E3, the Maryland case, and one entry per tool and question.
- `landscape.py`: the synthetic landscape, the Maryland case of E2 and E3,
  and the sites of E1 and the conductance of its edges, which every tool of E1
  shares. Smoke mode takes the top-left 32 x 32 pixels of Maryland.
- `install.sh`: installs the toolchains of the external tools and the data,
  each pinned: Julia and the packages of `julia/Manifest.toml`, through
  juliaup, the R packages of `renv.lock` in the renv library of this
  directory, which `Rscript` loads here through `.Rprofile`, and the land
  cover of Maryland in `data/`.
- `run.py`: the runner. Each cell runs in its own process, pinned with
  `taskset` to the cores of the configuration, with the OpenMP, BLAS, Julia
  and R thread variables matched to them; XLA sizes its thread pools to the
  same cores. GPU cells run on the configured GPU.
- `report.py`: the report, from a run directory.
- `adapters/`: one script per tool. `protocol.py` states the protocol between
  the runner and an adapter, `protocol.jl` and `protocol.R` implement it for
  Julia and R adapters, and `reference.py` computes the float64 reference of
  E1.
- `results/<run>/`: `records.jsonl`, the environment snapshot
  `environment.json`, the references of E1, the logs of the cells and the
  JAXScape maps of E2 and E3 (neither committed), and what the report writes: `results.csv`, one figure per
  question, `e1.png` to `e3.png`, and the page `index.md`.

## The cell record

A record is the one contract between the runner and the report:

| Field | Value |
| --- | --- |
| `question`, `tool`, `distance`, `device`, `dtype`, `solver`, `case` | The cell; `cells` is its number of raster cells. |
| `status`, `reason` | `ok`, `failed`, `timeout`, `oom` or `skipped`, and why. |
| `warmup_s`, `times_s`, `median_s` | The warm-up call, the timed calls and their median, in seconds. |
| `measure`, `agreement`, `tier` | The agreement measure of the tier, and its value, or `null` when the reference is unavailable. Tier 1 measures the largest relative error against the float64 reference of E1, and Tiers 2 and 3 Spearman's ρ against the JAXScape map of the same question and case. |
| `version`, `files`, `adapter` | The tool's version, whether its calls read or write files, and its adapter. |

## Add a tool

A tool takes one adapter script and one configuration entry.

1. Write `adapters/<tool>.py`, `.jl` or `.R`. Start it with a comment that
   states how it maps the shared parameters onto the tool; the report copies
   this comment into the table of parameter mappings. The adapter follows the
   protocol of `adapters/protocol.py`: it reads the cell file named by its
   argument, touches `progress` at the start of each call, saves the output
   of its last call to `output` (for E1, the distances among the sites, and
   for E2 and E3, the map, NaN on pixels off the tool's graph), and writes
   `result`. Julia adapters include `adapters/protocol.jl` and run
   with `julia --project=julia`. R adapters source `adapters/protocol.R` and
   run with `Rscript` in this directory. To add an R package, load it in the
   adapter, run `Rscript -e 'renv::install("samc"); renv::snapshot()'` here,
   and add it to `R_PACKAGES` in `run.py`, which records its version with each
   run.
2. Add an entry to `config.toml`, with the tool's settings for each distance.
   A setting given per device is a table keyed by device, and a list of
   solvers runs each. A tool of E2 or E3 is compared with the JAXScape map of
   the distance `compared_with`, by default its own, which the adapter
   justifies:

   ```toml
   [[tools]]
   tool = "Circuitscape.jl"
   question = "E1"
   adapter = "adapters/circuitscape.jl"
   devices = ["cpu"]
   tier = 1

   [tools.distances.resistance]
   solver = ["cg+amg", "cholmod"]
   ```

## Publish a run

`docs/benchmark` links to the published run directory, whose `index.md` is the
Benchmark page. To publish a run, report it and point the link at it:

```bash
uv run python report.py results/v0.0.16
ln -sfn ../benchmark/results/v0.0.16 ../docs/benchmark
```
