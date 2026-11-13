# The "Why JAXScape?" table

The "Why JAXScape?" section of `README.md` summarizes the benchmark: which
package covers which topic, and the speed-up of JAXScape over the fastest other
package on each topic. `docs_hooks.py` renders the README as the landing page of
the documentation, so the README is the only file to edit. Update the section
each time a benchmark run is published under `benchmark/results/<run>/` (see
"Publish a run" in `benchmark/README.md`).

## Steps

1. **Read the run.** Its page, `benchmark/results/<run>/index.md`, gives for
   each benchmark question the speed-up of JAXScape on the GPU over each tool,
   in the column "Speed-up" of the table under "Results", and defines it under
   "Timing": the warm runtime of the tool divided by that of JAXScape on the
   GPU, each in its fastest setting on the largest synthetic raster that both
   completed, with JAXScape under the same distance in the distance
   calculation, and under the distance or label of the column "Compared with
   JAXScape under" in the connectivity and sensitivity analyses. The page is
   authoritative. Map questions to columns: the distance calculation, by
   distance, to the three "Distance on a graph" columns; the connectivity
   analysis to "Connected habitat"; the sensitivity analysis to "Movement
   corridors and pinch points".
2. **Speed-up row.** For each column, take the speed-ups of the page over the
   other packages: in the distance calculation, those of the rows of the
   column's distance. The cell shows the smallest, as the page gives it (two
   significant digits below 10, an integer from 10), and names that package,
   e.g. `24x over Circuitscape.jl`. A speed-up below 1 stays as it is; add in
   parentheses why the package is faster when it solves a smaller problem
   (e.g. `on habitat patches`).
3. **Ticks.** A cell holds a plain tick `✓` when the package computes the
   quantity itself, and stays empty otherwise, a related quantity included. The
   ticks come from each package's own documentation; the benchmark runs only
   some packages on some topics. Change one only with a documentation link
   that supports it. Packages that cover none of the topics stay out of the
   table. Last check: 2026-11-12.
4. **Text.** Update the JAXScape version in the legend, which states the
   definition of the page. The section gives speed-ups only: absolute
   runtimes, rasters, failures (out of memory, errors, time limits) and
   agreement belong to the benchmark page, which the sentence after the legend
   links. Readers of the README do not have the benchmark details, so the
   table carries no other mark than the tick.
5. **Check.** `uv run mkdocs build --strict` passes, and the table fits the page
   width of the rendered landing page (packages are rows, topics are columns).
