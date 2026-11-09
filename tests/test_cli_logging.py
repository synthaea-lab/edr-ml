"""What a command-line entry point does with the package's log records.

Run in a fresh interpreter: pytest installs its own handlers on the root logger, which is
exactly the case where `configure_cli_logging` leaves the handlers alone.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_package_progress_goes_to_stderr_as_bare_lines_and_stdout_stays_the_output() -> None:
    result = _run(
        """
        import logging
        from synthaea_ml.cli_logging import configure_cli_logging

        configure_cli_logging()
        logging.getLogger("synthaea_ml.training.train_linux").info("Baseline: 3 entries")
        print("model written")
        """
    )

    assert result.stderr == "Baseline: 3 entries\n"
    assert result.stdout == "model written\n"


def test_third_party_info_stays_hidden_but_its_warnings_show() -> None:
    result = _run(
        """
        import logging
        from synthaea_ml.cli_logging import configure_cli_logging

        configure_cli_logging()
        logging.getLogger("numexpr.utils").info("NumExpr defaulting to 8 threads.")
        logging.getLogger("skl2onnx").warning("opset downgraded")
        """
    )

    assert result.stderr == "opset downgraded\n"


def test_without_the_cli_setup_library_progress_is_silent() -> None:
    result = _run(
        """
        import logging
        logging.getLogger("synthaea_ml.training.train_linux").info("Baseline: 3 entries")
        """
    )

    assert result.stderr == ""
    assert result.stdout == ""
