"""Logging setup for the command-line entry points.

Library functions report progress through `logging.getLogger(__name__)` and never write
to stdout themselves, so a caller can silence or route them. A command-line `main`
calls `configure_cli_logging` once: the package's INFO progress and every library's
warnings then reach stderr as plain lines, and stdout stays the command's own output.
"""

from __future__ import annotations

import logging

PACKAGE_LOGGER = "synthaea_ml"


def configure_cli_logging() -> None:
    """Send this package's INFO records and any library's warnings to stderr as bare messages.

    Third-party INFO chatter stays hidden: only the root threshold (WARNING) applies to it.
    A no-op on the handlers when the root logger already has one (an embedding application,
    or pytest), which keeps their routing; the package level is still raised to INFO.
    """
    logging.basicConfig(format="%(message)s", level=logging.WARNING)
    logging.getLogger(PACKAGE_LOGGER).setLevel(logging.INFO)
