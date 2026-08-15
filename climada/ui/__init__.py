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

Graphical user interface for physical climate risk assessment and adaptation
cost-benefit analysis.

The interface is a `Streamlit <https://streamlit.io>`_ application. It is an
optional component of CLIMADA; install its dependencies with::

    pip install climada[ui]

and launch it with either of::

    climada-ui
    python -m climada.ui

The modules :py:mod:`climada.ui.analysis`, :py:mod:`climada.ui.charts` and
:py:mod:`climada.ui.datasets` contain no Streamlit code and can be reused from
scripts and notebooks.
"""

from climada.ui.cli import main

__all__ = ["main"]
