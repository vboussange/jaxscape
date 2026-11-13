# The "Why JAXScape?" table

The "Why JAXScape?" section of `README.md` summarizes the benchmark: which
package covers which topic, and the speed-up of JAXScape over the fastest other
package on each topic. `docs_hooks.py` renders the README as the landing page of
the documentation, so the README is the only file to edit. Update the section
each time a benchmark run is published under `benchmark/results/<run>/` (see
"Publish a run" in `benchmark/README.md`).

## Steps

1. **Read the run.** Its page, `benchmark/results/<run>/index.md`, has one table
   per benchmark question, with the status of each cell (`ok`, `failed`,
   `timeout`, `oom`, `skipped`) or its warm runtime. Map questions to columns:
   the distance calculation, by distance, to the three "Effective distance"
   columns; the connectivity analysis to "Connected habitat"; the sensitivity
   analysis to "Movement corridors and pinch points". Use the synthetic
   rasters only, not the Maryland case.
2. **Speed-up row.** For each column and each other package, take the fastest
   configuration of each package (any distance, solver or device) and JAXScape
   on the GPU, on the largest raster on which both have status `ok`, and form
   the ratio runtime of the package / runtime of JAXScape. The cell shows the
   smallest ratio, with one decimal below 10 and as an integer from 10, and
   names that package, e.g. `24x over Circuitscape.jl`. A ratio below
   1 stays as it is; add in parentheses why the package is faster when it solves
   a smaller problem (e.g. `on habitat patches`).
3. **Ticks.** A cell holds a plain tick `✓` when the package computes the
   quantity itself, and stays empty otherwise, a related quantity included. The
   ticks come from each package's own documentation, not from the benchmark,
   which runs only some packages on some topics. Change one only with a
   documentation link that supports it. Packages that cover none of the topics
   stay out of the table. Last check: 2026-11-12.
4. **Text.** Update the JAXScape version in the legend. The section gives
   speed-ups only: absolute runtimes, rasters, failures (out of memory, errors,
   time limits) and agreement belong to the benchmark page, which the sentence
   after the legend links. Readers of the README do not have the benchmark
   details, so the table carries no other mark than the tick.
5. **Check.** `uv run mkdocs build --strict` passes, and the table fits the page
   width of the rendered landing page (packages are rows, topics are columns).
