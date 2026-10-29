# Repository guidance

Before changing code, tests, or documentation, read
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

Documentation, including docstrings, must state facts directly. Never use the
contrastive formula “This is X, not Y” or equivalent “X—not Y” phrasing.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues for vboussange/jaxscape, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root, created only when a term or decision gets settled. See `docs/agents/domain.md`.
