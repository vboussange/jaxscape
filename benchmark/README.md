# Benchmark

The benchmark compares JAXScape with the software that ecologists use for
connectivity analysis, on three questions: `distance`, the distance
calculation; `connectivity`, the connectivity analysis; and `sensitivity`, the
sensitivity analysis. The last two are windowed analyses. Its results make
the [Benchmark page](https://vboussange.github.io/jaxscape/benchmark/) of the
docs.

This directory is a uv project of its own, outside the root project, pinned to
the commit of a JAXScape release (`jaxscape @ git+...@<commit>` in
`pyproject.toml`, the commit of 0.0.18). Run every command from this
directory:

```bash
uv sync
./install.sh                                  # the pinned toolchains and the Maryland raster
uv run python run.py "$(mktemp -d)" --smoke   # every cell once, on 32 x 32 rasters, small windows
nohup uv run python run.py results/v0.0.18 > run.log 2>&1 &   # the full run
uv run python report.py results/v0.0.18      # the figures and the page
uv run pytest                                 # tests of the runner and the report
```

Smoke mode runs in a new run directory, such as a scratch directory, and exits
with an error unless every cell is `ok` with its agreement and every Tier 1
error is at most `smoke_tolerance`, which catches a tool whose graph departs
from the shared edge rule. A full run appends one record per cell to
`records.csv` as each cell finishes; when it restarts, it skips the cells
recorded there. To benchmark a later release, change the pin, run, and report.

## Layout

- `config.toml`: how cells run (timeout, repeats, cores, GPU, smoke mode), the
  landscape, the raster sides and shared parameters of each distance, the
  windows of the windowed analyses, the Maryland case, one entry per tool and
  question, and the solver view, which runs JAXScape with each solver in
  float32 and float64 on the GPU: the distance calculation on every raster of
  its distance, the windowed analyses on exact resistance distance at 256²
  alone, and Spielman's approximation of resistance distance on every case.
- `landscape.py`: the synthetic landscape, the Maryland case of the windowed
  analyses, the sites of the distance calculation and the conductance of its
  edges, which every tool of the distance calculation shares, and the
  resistance raster of the adapters. Smoke mode takes the top-left 32 x 32
  pixels of Maryland.
- `install.sh`: installs the toolchains of the external tools and the data,
  each pinned: Julia and the packages of `julia/Manifest.toml`, through
  juliaup, the R packages of `renv.lock` in the renv library of this
  directory, which `Rscript` loads here through `.Rprofile`, the Graphab jar
  and the Java runtime that it needs in `java/`, and the land cover of
  Maryland in `data/`. The Python tools are dependencies of this project.
- `run.py`: the runner. Each cell runs in its own process, pinned with
  `taskset` to the cores of the configuration, with the OpenMP, BLAS, Julia
  and R thread variables matched to them; XLA sizes its thread pools to the
  same cores. GPU cells run on the configured GPU, numbered as `nvidia-smi`
  numbers it.
- `report.py`: the report, from a run directory.
- `adapters/`: one script per tool. `protocol.py` states the protocol between
  the runner and an adapter, `protocol.jl` and `protocol.R` implement it for
  Julia and R adapters, and `reference.py` computes the float64 reference of
  the distance calculation.
- `results/<run>/`: `records.csv`, the configuration of the run `config.json`,
  smoke settings applied, the environment snapshot `environment.json`, the
  references of each question, the logs of the cells and the maps of the
  windowed analyses (neither committed): those of JAXScape's GPU cells, and
  the other tools' maps of Maryland. The report writes one figure per
  question, `distance.png`, `connectivity.png` and `sensitivity.png`, the
  Maryland figure `maryland.png`, which it draws when the run directory holds
  its maps, the figures of the solver view, `solvers.png` and
  `solver-errors.png`, and the page `index.md`, whose settings come from
  `config.json`.

## The cell record

A record, one row of `records.csv`, is the one contract between the runner and
the report. `FIELDS` in `run.py` fixes the header; the times of the timed calls
make one value, joined by `;`, and a missing value is empty.

| Field | Value |
| --- | --- |
| `view`, `question`, `tool`, `distance`, `device`, `dtype`, `solver`, `case` | The cell, in the `comparison` or the `solver` view; `cells` is its number of raster cells. The `distance` of a configuration whose table is keyed by a label is the label. |
| `status`, `reason` | `ok`, `failed`, `timeout`, `oom` or `skipped`, and why. |
| `warmup_s`, `times_s`, `median_s` | The warm-up call, the timed calls and their median, in seconds. |
| `measure`, `agreement`, `tier`, `compared_with` | The agreement measure of the tier, or of the solver view, and its value, or missing when the reference is unavailable or the measure undefined. At Tier 1, the measure is the largest relative error with respect to the float64 reference of the distance calculation, and at Tiers 2 and 3, Spearman's ρ between the map of the tool and the map of JAXScape's GPU cell of the same question and case, under the distance, or the label, `compared_with`. In the solver view, the measure is the largest relative error with respect to the reference of the distance calculation, and in the windowed analyses the relative L2 error with respect to the map of exact resistance distance in float64, on the cases of the question's distance alone: elsewhere, `measure` is missing. The page does not show `total`. |
| `total` | The landscape total of the connectivity analysis, $\sum_i q_i S_i$, of a tool that computes one: JAXScape's `ConnectivityAnalysis`, and the square of Graphab's equivalent connectivity. |
| `version`, `files`, `adapter` | The tool's version, from the environment snapshot, whether its calls read or write files, and its adapter. |

## Add a tool

A tool takes one adapter script and one configuration entry.

1. Write `adapters/<tool>.py`, `.jl` or `.R`. Start it with a comment that
   states how it maps the shared parameters onto the tool; the report copies
   this comment into the table of parameter mappings. The adapter follows the
   protocol of `adapters/protocol.py`: it reads the cell file named by its
   argument, touches `progress` at the start of each call, saves the output
   of its last call to `output` (for the distance calculation, the distances
   among the sites, and for the windowed analyses, the map, NaN on pixels off
   the tool's graph), and writes `result`. Julia adapters include
   `adapters/protocol.jl` and run with `julia --project=julia`. R adapters
   source `adapters/protocol.R` and run with `Rscript` in this directory. To
   add an R package, load it in the adapter, and run
   `Rscript -e 'renv::install("samc"); renv::snapshot()'` here.
2. Add an entry to `config.toml`, with the tool's settings for each distance.
   A setting given per device is a table keyed by device, and a list of
   solvers runs each. A table keyed by a label of its own computes under the
   distance of its `distance` setting, and its records carry the label in
   place of the distance, as JAXScape's `resistance, K = -R, normalized` of
   the sensitivity analysis. A tool of a windowed analysis is compared with
   the map of JAXScape's GPU cell under the distance, or the label,
   `compared_with`, by default its own distance, which the adapter justifies.
   `package` names the tool in the pins of its toolchain, which give the
   version of its records, when it differs from the tool's name without
   `.jl`:

   ```toml
   [[tools]]
   tool = "Circuitscape.jl"
   question = "distance"
   adapter = "adapters/circuitscape.jl"
   devices = ["cpu"]
   tier = 1

   [tools.distances.resistance]
   solver = ["cg+amg", "cholmod"]
   ```

## Add a solver

A solver of the solver view takes one entry in `[solver_view.solvers]` of
`config.toml`: its label, which its records carry as `solver`, and the
settings that it adds or replaces, such as a `method` of resistance. It runs
in each dtype on each distance of each question of the solver view, unless its
`questions` and `dtypes` restrict it, or an entry of `fails_by_design` matches
it. `cases` restricts the cases on which it runs, and `timeout` replaces that
of the view:

```toml
[solver_view.solvers]
GeometricMultigridCGSolver = {}
"GeometricMultigridCGSolver(rtol=1e-3)" = { dtypes = ["float32"] }
```

## Publish a run

`docs/benchmark` links to the published run directory, whose `index.md` is the
Benchmark page. To publish a run, report it with `--publish`, which points the
link at it, and commit both: the link, and the run's `records.csv`,
`config.json`, `environment.json`, figures and page. Its logs, maps,
references and `run.log` stay out of the repository.

```bash
uv run python report.py results/v0.0.18 --publish
```
