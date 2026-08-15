"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Command line launcher for the CLIMADA user interface.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

__all__ = ["main"]

APP_PATH = Path(__file__).parent / "app.py"

THEME_PRIMARY = "#2a78d6"
"""Slot 1 of the chart palette, reused for the interface's own accents."""

INSTALL_HINT = (
    "The CLIMADA user interface needs Streamlit and Plotly, which are optional "
    "dependencies. Install them with:\n\n    pip install climada[ui]\n"
)


def build_parser() -> argparse.ArgumentParser:
    """The ``climada-ui`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="climada-ui",
        description=(
            "Launch the CLIMADA interface for physical risk assessment and "
            "adaptation cost-benefit analysis."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to serve on. Default: 8501.",
    )
    parser.add_argument(
        "--address",
        default="localhost",
        help="Address to bind to. Default: localhost.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a browser window.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Start the Streamlit server hosting the interface.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. Default: ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        # Imported lazily: Streamlit is an optional dependency, and the
        # message below is more useful than an ImportError traceback.
        # pylint: disable=import-outside-toplevel
        from streamlit.web import cli as stcli
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    sys.argv = [
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true" if args.headless else "false",
        "--browser.gatherUsageStats",
        "false",
        # Keep the interface chrome on the same palette as the charts, so a
        # primary button never reads as a status colour.
        "--theme.primaryColor",
        THEME_PRIMARY,
        "--theme.font",
        "sans-serif",
    ]
    return int(stcli.main() or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
