# Contributing

Pull requests are welcome. This guide follows
[Equinox's contribution workflow](https://github.com/patrick-kidger/equinox/blob/main/CONTRIBUTING.md),
with setup commands adapted to JAXScape.

---

### Getting started

Install [uv](https://docs.astral.sh/uv/) and fork JAXScape on GitHub. Clone your
fork, install the project with its lint, test and documentation dependencies,
and enable the repository's pre-commit hooks:

```bash
git clone https://github.com/your-username-here/jaxscape.git
cd jaxscape
uv sync --extra lint --extra tests --extra docs --extra amjax --extra cholespy
uv run pre-commit install
```

The hooks format and lint the code with ruff and type-check it with pyright, at
the versions of the `lint` extra, in the project environment. They check the
files of each commit; pyright checks the library and the tests as a whole.

---

### Code changes

Make your changes and update tests where needed. Run the existing test suite,
with one pytest-xdist worker per core:

```bash
uv run pytest -n auto tests
```

On Linux, each worker runs on a core of its own, and `pytest tests` runs the
tests one after another. The default run leaves out the slow tests, marked
`slow`, and those marked `gpu`, which need an NVIDIA GPU with the `cudss`
extra. The continuous integration runs every test but the GPU ones:

```bash
uv run pytest -n 2 -m slow tests          # slow tests only
uv run pytest -n auto -m "not gpu" tests  # every test but the GPU ones, as in CI
uv run pytest -m gpu tests                # GPU tests only
uv run pytest -n auto -m "" tests         # every test
```

The machines of the continuous integration have no NVIDIA GPU. They test the
logic of `AbstractSparseDirectSolver` with a dense Cholesky factorization in
place of cuDSS, and the `CuDSSSolver` tests, marked `gpu`, run only on a
machine with an NVIDIA GPU, spineax and cuDSS.

The continuous integration leaves out the tests of the benchmark, a uv project
of its own in `benchmark/`; run them from that directory with `uv sync` and
`uv run pytest`, as `benchmark/README.md` describes.

Before opening a pull request, run every hook on every file, as the
continuous integration does:

```bash
uv run pre-commit run --all-files
```

Commit your changes, push them to your fork, and open a pull request:

```bash
git push
```

---

### Class design

Follow Equinox's coding practices, including its
[abstract/final pattern](https://docs.kidger.site/equinox/pattern/): every class
is either abstract or final.

- An abstract class is named `Abstract*` and is meant to be subclassed, as
  `AbstractDistance` is.
- A final class is concrete and is never subclassed, as `LCPDistance` is.

The rule applies to new code. Some existing classes predate it and have yet to
follow it.

---

### Documentation changes

Use Markdown docstrings with a short summary, `**Arguments:**` (or
`**Attributes:**` for class configuration), and `**Returns:**` where applicable.
List parameter names in backticks and describe shapes, constraints, and useful
defaults. Use fenced `python` examples and lowercase MkDocs admonitions such as
`!!! note`.

```python
"""Compute pairwise distances.

**Arguments:**

- `nodes`: Vertex indices with shape `(m,)`.

**Returns:**

Distance matrix with shape `(m, m)`.
"""
```

Edit the documentation, then start the local preview:

```bash
uv run mkdocs serve
```

Open <http://localhost:8000> to check the rendered pages before submitting your
pull request. Run the same strict build as CI; warnings fail the build:

```bash
uv run mkdocs build --strict
```

The build renders the outputs stored in the example notebooks without
executing them. After changing a notebook, execute it and commit it with its
outputs. Set `TQDM_DISABLE=1`, otherwise every update of a progress bar is
stored, and rendered, as a separate output:

```bash
TQDM_DISABLE=1 uv run --with nbconvert --with ipykernel --with matplotlib \
  --with rasterio jupyter nbconvert --to notebook --execute --inplace \
  docs/examples/<name>/<name>.ipynb
```

`README.md` is also the landing page of the documentation: `docs_hooks.py`
renders it in place of `docs/index.md`, so edit the README only.
