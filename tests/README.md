# ml/tests

Unit tests for the package and the cross-language parity fixtures: `fixtures/` holds
the event/feature-vector pairs that BOTH this package and `crates/ml` assert against.
Changing a fixture is a coordinated two-sided change by definition.
The scripts that regenerate them live in `fixtures/generators/` (see
`fixtures/README.md`).
