# Development

How to set up a working copy, which checks a change must pass, and the conventions the code
follows. The repository's [CONTRIBUTING.md](../CONTRIBUTING.md) is the short version; the
organization-wide [contributing guide](https://github.com/synthaea-lab/.github/blob/main/CONTRIBUTING.md)
covers issues, the Developer Certificate of Origin and licensing.

## Setup

Python 3.11 or later. The `dev` extra adds pinned versions of ruff, mypy and the PyYAML stubs,
and pytest:

```bash
git clone https://github.com/synthaea-lab/ml
cd ml
python3 -m venv .venv
source .venv/bin/activate          # Windows: py -m venv .venv, then .venv\Scripts\activate
pip install -e ".[dev]"
```

For the parity and provenance tests, clone the agent and scenarios repositories next to it
([testing.md](testing.md#sibling-checkouts)), or point `SYNTHAEA_ROOT` at their parent.

### Dependencies

| Dependency | Why |
| --- | --- |
| numpy | Feature matrices |
| scikit-learn | `IsolationForest`, the calibration split, ROC AUC |
| skl2onnx | Export to ONNX |
| onnxruntime | `verify_onnx`, the shipped-model adapter of the global floor |
| PyYAML | Scenario sidecars |

The feature extractors and the dataset readers use only the standard library; the heavier
dependencies serve training, export and verification. The attribution reference in the tests
also uses `onnx`, which skl2onnx brings in.

## Checks

From the repository root:

```bash
ruff check .
mypy
pytest tests
```

All three must pass with the siblings present and `SYNTHAEA_REQUIRE_SIBLINGS=1`.

- **ruff** (line length 100) enables, beyond the defaults, bugbear, simplifications,
  comprehensions, complexity (McCabe 10) and argument-count limits, unused arguments,
  commented-out code, security (`S`), `print` in library code, import order, upgrades,
  timezone-aware datetimes and docstrings on the public API (`D1`).
- **mypy** runs in strict mode on `synthaea_ml/`, with `warn_unreachable`. scikit-learn, skl2onnx,
  onnxruntime and onnx ship no types and are ignored.

### Ratchets

`[tool.ruff.lint.per-file-ignores]` and the `[[tool.mypy.overrides]]` blocks in
[`pyproject.toml`](../pyproject.toml) list the known exceptions per module: the command-line
modules may print, a few modules still lack docstrings or complete type arguments. Both are
ratchets. An entry only ever shrinks or leaves; a new finding is fixed, not added to the table.

## Continuous integration

[`.github/workflows/ml.yml`](../.github/workflows/ml.yml) runs on every pull request and on
pushes to `main`, with a read-only token:

1. Check out this repository as `ml/`, and the agent (`crates/ml/tests/fixtures` only) and
   scenarios (`scenarios/` and `ml-capture/` only) repositories beside it.
2. Set up Python 3.12 and `pip install -e ".[dev]"`.
3. Run ruff, mypy and pytest with `SYNTHAEA_ROOT` set to the directory holding the three checkouts and
   `SYNTHAEA_REQUIRE_SIBLINGS=1`.

The job runs on Ubuntu, Windows and macOS, without fail-fast, because developers train and test
on all three. Every action is pinned by commit SHA.

[`.github/workflows/dco.yml`](../.github/workflows/dco.yml) requires a `Signed-off-by` line on
every commit of a pull request: commit with `git commit -s`.

## Conventions

### Command-line conventions

- Each entry point module defines `main(argv: Sequence[str] | None = None) -> int`, usually as
  `parse_args` into a frozen dataclass, then `run(args)`. `if __name__ == "__main__":
  sys.exit(main())` runs it.
- Each `main` is registered under `[project.scripts]` as `synthaea-ml-<name>`.
  `test_cli_entry_points.py` fails if one is missing.
- Library code logs through `logging.getLogger(__name__)` and never prints. A `main` calls
  [`configure_cli_logging()`](../synthaea_ml/cli_logging.py) once: the package's INFO records and
  any library's warnings go to stderr as bare messages, and stdout stays the command's own
  output. The ruff `T201` exceptions are limited to the command-line modules.

### Data handling

- Read JSON Lines through [`iter_jsonl`](../synthaea_ml/data/jsonl.py) and state the bad-line
  policy explicitly.
- Build model input from an exec record through [`data/canonical.py`](../synthaea_ml/data/canonical.py),
  never from a display string.
- Hash files with [`hashing.sha256_file`](../synthaea_ml/hashing.py).
- Write timestamps as timezone-aware UTC; ruff's `DTZ` rules reject naive datetimes.
- Put a value two modules must agree on in the lowest layer that both import:
  `training/config.py` for hyperparameters and thresholds, `features/sanity_samples.py` for the
  sanity samples, `registry/layout.py` for file names.

### Exploratory work

Notebooks and one-off scripts stay out of the repository until they matter. Anything
load-bearing moves into `synthaea_ml/` with tests; nothing exploratory is imported by the package
or run by the checks.

## Coordinated changes

Some changes cannot land in this repository alone:

| Change | Also needed |
| --- | --- |
| A feature definition, the canonical command line or a wire field the features read | The agent's Rust twin, regenerated goldens, a retrain ([features.md](features.md#changing-a-feature)) |
| A scenario sidecar a model record binds | A retrain of every model bound to it; the scenarios repository requires it |
| The `model_record.json` or manifest format | A schema version bump; readers reject unknown versions |
| `model_metadata.json` | The agent's scorer, which reads it |

Open the pull requests together and reference each other, so neither side merges into a state the
other cannot test against.
