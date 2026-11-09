# Contributing to Synthaea ML

This guide covers what is specific to this repository. The organization-wide
[contributing guide](https://github.com/synthaea-lab/.github/blob/main/CONTRIBUTING.md) states
the rules that hold everywhere: issues, the Developer Certificate of Origin, licensing and
conduct.

## Development setup

Install the package with the `dev` extra, which pins the linter and type-checker versions the
checks are run with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, use `py -m venv .venv` and `.venv\Scripts\activate`.

### Sibling checkouts

Some tests read files owned by other public repositories: the Rust parity fixtures in
[agent](https://github.com/synthaea-lab/agent) (`crates/ml/tests/fixtures`) and the scenario
definitions and capture scripts in [scenarios](https://github.com/synthaea-lab/scenarios)
(`scenarios/`, `ml-capture/`). Model records bind to scenario files by content hash, so
provenance checks need them too. Clone them next to this repository:

```bash
cd ..
git clone https://github.com/synthaea-lab/agent
git clone https://github.com/synthaea-lab/scenarios
```

To keep them elsewhere, set `SYNTHAEA_ROOT` to the directory that contains the `agent/` and
`scenarios/` checkouts. A test whose sibling is missing is skipped; set
`SYNTHAEA_REQUIRE_SIBLINGS=1` to make it fail instead, which is how the checks run before a
merge.

## Checks

Run from the repository root:

```bash
ruff check .
mypy
pytest tests
```

All three must pass, with the siblings present and `SYNTHAEA_REQUIRE_SIBLINGS=1`, on Linux,
Windows and macOS. Lint and type-check exceptions in `pyproject.toml` only ever shrink: fix
the finding instead of adding an entry.

## Changing features and fixtures

A feature definition exists twice, in `synthaea_ml/features/` and in the agent's `ml` crate,
and both test against the golden data in `tests/fixtures/`. Changing a feature, a model recipe
or a wire format is therefore a coordinated change to both repositories: regenerate the
affected goldens with the scripts in `tests/fixtures/generators/` (see the
[fixtures README](tests/fixtures/README.md)) and open the matching pull request on the agent.
Never regenerate a golden to make a failing parity test pass.

## Models and datasets

- A new registry version needs a model card and the `model_record.json` its trainer writes.
  `python -m synthaea_ml.export.verify_onnx` must pass for it with
  `SYNTHAEA_STRICT_PROVENANCE=1`.
- A new baseline goes under `datasets/baselines/` with a `manifest.json` written by
  `python -m synthaea_ml.data.manifest write`. Raw captures and labeled traces are not
  committed.

## Exploratory work

Notebooks and one-off scripts stay out of this repository until they matter. Anything
load-bearing graduates into `synthaea_ml/` with tests; nothing exploratory is imported by the
package or run by the checks.
