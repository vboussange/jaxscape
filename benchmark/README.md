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
and exits with an error unless every cell is `ok` with its agreement and every
Tier 1 error is at most `smoke_tolerance`, which catches a tool whose graph
departs from the shared edge rule. A full run appends one record per cell to
`records.jsonl` as each cell finishes; when it restarts, it skips the cells
recorded there. To benchmark a later release, change the pin, run, and report.

## Layout

- `config.toml`: how cells run (timeout, repeats, cores, GPU, smoke mode), the
  landscape, the raster sides and shared parameters of each distance, the
  windows of E2 and E3, the Maryland case, one entry per tool and question,
  and the solver view, which runs JAXScape with each solver in float32 and
  float64 on the GPU: E1 on every raster of its distance, and E2 and E3 on
  256² alone.
- `landscape.py`: the synthetic landscape, the Maryland case of E2 and E3, the
  sites of E1 and the conductance of its edges, which every tool of E1 shares,
  and the resistance raster of the adapters. Smoke mode takes the top-left
  32 x 32 pixels of Maryland.
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
  E1.
- `results/<run>/`: `records.jsonl`, the configuration of the run
  `config.json`, smoke settings applied, the environment snapshot
  `environment.json`, the references of E1, E2 and E3, the logs of the cells
  and the maps of E2 and E3 (neither committed): those of JAXScape's GPU
  cells, and the other tools' maps of Maryland. The report writes
  `results.csv`, one figure per question, `e1.png` to `e3.png`, the Maryland
  figure `maryland.png`, which it draws when the run directory holds its maps,
  the figure of the solver view, `solvers.png`, and the page `index.md`, whose
  settings come from `config.json`. `notes.toml`, written by hand, holds notes
  on the run, such as why a cell failed: Markdown keyed by a question, such as
  `E1`, or by `Solvers`, which the page adds after the table of that section.

## The cell record

A record is the one contract between the runner and the report:

| Field | Value |
| --- | --- |
| `view`, `question`, `tool`, `distance`, `device`, `dtype`, `solver`, `case` | The cell, in the `comparison` or the `solver` view; `cells` is its number of raster cells. The `distance` of a configuration whose table is keyed by a label is the label. |
| `status`, `reason` | `ok`, `failed`, `timeout`, `oom` or `skipped`, and why. |
| `warmup_s`, `times_s`, `median_s` | The warm-up call, the timed calls and their median, in seconds. |
| `measure`, `agreement`, `tier`, `compared_with` | The agreement measure of the tier, or of the solver view, and its value, or `null` when the reference is unavailable or the measure undefined. Tier 1 measures the largest relative error against the float64 reference of E1, and Tiers 2 and 3 Spearman's ρ against the map of JAXScape's GPU cell of the same question and case, under the distance, or the label, `compared_with`. The solver view measures the largest relative error against the reference of E1 for E1, and for E2 and E3 the relative L2 error against the map of exact resistance in float64. |
| `total` | The landscape total of E2, $\sum_i q_i S_i$, of a tool that computes one: JAXScape's `ConnectivityAnalysis`, and the square of Graphab's equivalent connectivity. |
| `version`, `files`, `adapter` | The tool's version, from the environment snapshot, whether its calls read or write files, and its adapter. |

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
   adapter, and run `Rscript -e 'renv::install("samc"); renv::snapshot()'`
   here.
2. Add an entry to `config.toml`, with the tool's settings for each distance.
   A setting given per device is a table keyed by device, and a list of
   solvers runs each. A table keyed by a label of its own computes under the
   distance of its `distance` setting, and its records carry the label in
   place of the distance, as JAXScape's `resistance, K = -R, normalized` of
   E3. A tool of E2 or E3 is compared with the map of JAXScape's GPU cell
   under the distance, or the label, `compared_with`, by default its own
   distance, which the adapter justifies. `package` names the tool in the
   pins of its toolchain, which give the version of its records, when it
   differs from the tool's name without `.jl`:

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

## Add a solver

A solver of the solver view takes one entry in `[solver_view.solvers]` of
`config.toml`: its label, which its records carry as `solver`, and the
settings that it adds, such as a `method` of resistance. It runs in each dtype
on each distance of each question of the solver view, unless its `questions`
and `dtypes` restrict it, or an entry of `fails_by_design` matches it:

```toml
[solver_view.solvers]
GridMultigridSolver = {}
"GridMultigridSolver(rtol=1e-3)" = { dtypes = ["float32"] }
```

## Publish a run

`docs/benchmark` links to the published run directory, whose `index.md` is the
Benchmark page. To publish a run, report it with `--publish`, which points the
link at it, and commit both: the link, and the run's `records.jsonl`,
`config.json`, `environment.json`, `notes.toml`, `results.csv`, figures and
page. Its logs, maps, references and `run.log` stay out of the repository.

```bash
uv run python report.py results/v0.0.16 --publish
```
