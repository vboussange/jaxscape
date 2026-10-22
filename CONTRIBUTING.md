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

Make your changes and update tests where needed. Run the existing test suite:

```bash
uv run pytest tests
```

Commit your changes, push them to your fork, and open a pull request:

```bash
git push
```

---

### Documentation changes

Edit the documentation, then start the local preview:

```bash
uv run mkdocs serve
```

Open <http://localhost:8000> to check the rendered pages before submitting your
pull request.
