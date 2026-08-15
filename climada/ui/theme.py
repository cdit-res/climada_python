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

Colour tokens and Plotly template for the CLIMADA user interface.

The categorical palette is validated for colour-vision deficiency on the
adjacent-pair list in both light and dark mode. Slots are always assigned in
the order given here and never cycled: beyond
:py:data:`MAX_CATEGORICAL_SERIES` series the caller must aggregate into an
"other" bucket or facet the chart.
"""

from typing import Dict, List

import plotly.graph_objects as go

__all__ = [
    "MAX_CATEGORICAL_SERIES",
    "SEQUENTIAL_BLUE",
    "STATUS",
    "categorical",
    "plotly_template",
    "sequential_scale",
    "tokens",
]


CATEGORICAL_LIGHT: List[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
"""Categorical slots for the light chart surface, in fixed assignment order."""

CATEGORICAL_DARK: List[str] = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
"""Categorical slots for the dark chart surface -- the same eight hues, restepped."""

MAX_CATEGORICAL_SERIES = len(CATEGORICAL_LIGHT)
"""Number of distinct categorical slots available before folding into 'other'."""

SEQUENTIAL_BLUE: List[str] = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
"""Single-hue ramp, light to dark, for continuous magnitude encodings."""

STATUS: Dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
"""Reserved state colours. Never reused as a series colour, always paired with a label."""

_LIGHT_TOKENS: Dict[str, str] = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "positive": "#006300",
}

_DARK_TOKENS: Dict[str, str] = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "positive": "#0ca30c",
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def tokens(dark: bool = False) -> Dict[str, str]:
    """Chart chrome and ink colours for the requested mode.

    Parameters
    ----------
    dark : bool, optional
        Return the dark-surface tokens instead of the light ones. Default: False.

    Returns
    -------
    dict
        Mapping of role name to colour.
    """
    return dict(_DARK_TOKENS if dark else _LIGHT_TOKENS)


def categorical(dark: bool = False) -> List[str]:
    """Categorical palette for the requested mode, in fixed slot order.

    Parameters
    ----------
    dark : bool, optional
        Return the dark-surface steps instead of the light ones. Default: False.

    Returns
    -------
    list of str
    """
    return list(CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT)


def sequential_scale(dark: bool = False) -> List[str]:
    """Single-hue ramp for continuous magnitude, oriented for the given surface.

    On a dark surface the ramp is reversed so that "near zero" is the step
    closest to the background in both modes.

    Parameters
    ----------
    dark : bool, optional
        Orient the ramp for the dark surface. Default: False.

    Returns
    -------
    list of str
    """
    return list(reversed(SEQUENTIAL_BLUE)) if dark else list(SEQUENTIAL_BLUE)


def plotly_template(dark: bool = False) -> go.layout.Template:
    """Plotly template carrying the CLIMADA UI chart chrome.

    Grid and axis lines are recessive hairlines, the surface is explicit (so the
    figure never borrows a transparent background), and the categorical
    colourway is the validated fixed-order palette.

    Parameters
    ----------
    dark : bool, optional
        Build the dark-surface template. Default: False.

    Returns
    -------
    plotly.graph_objects.layout.Template
    """
    tok = tokens(dark)
    axis = {
        "gridcolor": tok["grid"],
        "linecolor": tok["axis"],
        "zerolinecolor": tok["axis"],
        "tickcolor": tok["axis"],
        "tickfont": {"color": tok["text_muted"], "size": 12},
        "title": {"font": {"color": tok["text_secondary"], "size": 13}},
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "colorway": categorical(dark),
            "paper_bgcolor": tok["surface"],
            "plot_bgcolor": tok["surface"],
            "font": {
                "family": FONT_FAMILY,
                "color": tok["text_primary"],
                "size": 13,
            },
            "title": {
                "font": {"color": tok["text_primary"], "size": 16},
                "x": 0,
                "xanchor": "left",
            },
            "xaxis": axis,
            "yaxis": axis,
            "legend": {
                "font": {"color": tok["text_secondary"], "size": 12},
                "bgcolor": "rgba(0,0,0,0)",
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            "hoverlabel": {"font": {"family": FONT_FAMILY, "size": 12}},
            "margin": {"l": 60, "r": 24, "t": 56, "b": 48},
            "colorscale": {"sequential": _as_colorscale(sequential_scale(dark))},
        }
    )


def _as_colorscale(colors: List[str]):
    """Turn an ordered colour list into a Plotly colourscale."""
    if len(colors) == 1:
        return [(0.0, colors[0]), (1.0, colors[0])]
    step = 1.0 / (len(colors) - 1)
    return [(min(1.0, i * step), color) for i, color in enumerate(colors)]
