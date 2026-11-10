# Contributing

Pull requests are welcome. This guide follows
[Equinox's contribution workflow](https://github.com/patrick-kidger/equinox/blob/main/CONTRIBUTING.md),
with setup commands adapted to JAXScape.

---

### Getting started

Install [uv](https://docs.astral.sh/uv/) and fork JAXScape on GitHub. Clone your
fork, install the project with its test and documentation dependencies, and
enable the repository's pre-commit hooks:

```bash
git clone https://github.com/your-username-here/jaxscape.git
cd jaxscape
uv sync --extra tests --extra docs --extra pyamg --extra amjax --extra cholespy
uvx pre-commit install
```

---

### Code changes

Make your changes and update tests where needed. Run the existing test suite,
with one pytest-xdist worker per core:

```bash
uv run pytest -n auto tests
```

On Linux, each worker runs on a core of its own, and `pytest tests` runs the
tests one after another. The default run leaves out the tests marked `slow`,
which take up to a minute each on one core, and those marked `gpu`, which need
an NVIDIA GPU with the `cudss` extra. The continuous integration runs every
test but the GPU ones:

```bash
uv run pytest -n auto -m slow tests       # slow tests only
uv run pytest -n auto -m "not gpu" tests  # every test but the GPU ones, as in CI
uv run pytest -m gpu tests                # GPU tests only
uv run pytest -n auto -m "" tests         # every test
```

To keep the compiled programs between runs, as the continuous integration
does, set `JAX_COMPILATION_CACHE_DIR` to a directory and
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`: later runs compile only the
programs that changed.

Commit your changes, push them to your fork, and open a pull request:

```bash
git push
```

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
