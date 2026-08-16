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

Plotly figures for the CLIMADA user interface.

Every figure here is a pure function of data plus a ``dark`` flag; nothing
reaches into Streamlit. Each chart the interface shows is accompanied by the
underlying table in the view layer, which is what licenses the lighter
categorical steps on the light surface.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from climada.ui.formatting import norm_values
from climada.ui.theme import (
    MAX_CATEGORICAL_SERIES,
    STATUS,
    categorical,
    plotly_template,
    sequential_scale,
    tokens,
)

__all__ = [
    "cost_benefit_chart",
    "exceedance_curve",
    "impact_function_chart",
    "measure_bcr_bars",
    "point_map",
    "residual_risk_bars",
    "return_period_bars",
    "series_bar_chart",
    "waterfall_chart",
]

_CORNER_RADIUS = 4
"""Rounded data-end radius, in pixels, for bar marks."""

_LINE_WIDTH = 2


def _base_figure(dark: bool, title: str = "", height: int = 380) -> go.Figure:
    """A figure carrying the UI template and a left-aligned title."""
    figure = go.Figure()
    figure.update_layout(
        template=plotly_template(dark), height=height, title_text=title
    )
    return figure


def _axis_scale(values: Sequence[float], unit: str) -> Tuple[float, str]:
    """Divisor and axis label suffix for a set of values."""
    finite = [value for value in np.ravel(values) if np.isfinite(value)]
    factor, name = norm_values(max(finite) if finite else 0.0)
    suffix = " ".join(part for part in (name, unit) if part)
    return factor, suffix


_LOG_TICK_CANDIDATES = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _log_ticks(values: Sequence[float]) -> Dict[str, object]:
    """Readable tick marks for a logarithmic return-period axis.

    Plotly's automatic log ticks label decade subdivisions with bare mantissas
    ("2" for 20), which reads as a smaller number than it is. Naming the ticks
    explicitly avoids that.

    Parameters
    ----------
    values : sequence of float
        Every value plotted on the axis.

    Returns
    -------
    dict
        Keyword arguments for ``update_xaxes``.
    """
    finite = [value for value in values if value and np.isfinite(value) and value > 0]
    if not finite:
        return {}
    low, high = min(finite), max(finite)
    ticks = [tick for tick in _LOG_TICK_CANDIDATES if low <= tick <= high]
    if len(ticks) < 2:
        return {}
    return {
        "tickmode": "array",
        "tickvals": ticks,
        "ticktext": [f"{tick:,g}" for tick in ticks],
    }


def exceedance_curve(
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    unit: str = "",
    dark: bool = False,
    title: str = "Impact exceedance frequency curve",
) -> go.Figure:
    """Loss exceedance curves on a logarithmic return-period axis.

    Parameters
    ----------
    series : dict
        Mapping of label to ``(return_periods, impacts)``. Series are assigned
        categorical slots in insertion order.
    unit : str, optional
        Value unit, used on the y-axis label.
    dark : bool, optional
        Render for the dark surface. Default: False.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    palette = categorical(dark)
    all_impacts = [value for _, impacts in series.values() for value in impacts]
    factor, suffix = _axis_scale(all_impacts, unit)

    for index, (label, (return_per, impacts)) in enumerate(series.items()):
        return_per = np.asarray(return_per, dtype=float)
        impacts = np.asarray(impacts, dtype=float)
        keep = return_per > 0
        figure.add_trace(
            go.Scatter(
                x=return_per[keep],
                y=impacts[keep] / factor,
                name=label,
                mode="lines",
                line={
                    "width": _LINE_WIDTH,
                    "color": palette[index % MAX_CATEGORICAL_SERIES],
                },
                hovertemplate=(
                    f"<b>{label}</b><br>Return period %{{x:,.0f}} y"
                    f"<br>Impact %{{y:,.3g}} {suffix}<extra></extra>"
                ),
            )
        )

    figure.update_layout(hovermode="x unified", showlegend=len(series) > 1)
    figure.update_xaxes(
        type="log",
        title_text="Return period (years)",
        showgrid=True,
        **_log_ticks([rp for return_per, _ in series.values() for rp in return_per]),
    )
    figure.update_yaxes(title_text=f"Impact ({suffix})".strip(), rangemode="tozero")
    return figure


def return_period_bars(
    return_periods: Sequence[float],
    impacts: Sequence[float],
    unit: str = "",
    dark: bool = False,
    title: str = "Loss by return period",
) -> go.Figure:
    """Losses at selected return periods, as bars with direct labels.

    Parameters
    ----------
    return_periods : sequence of float
    impacts : sequence of float
    unit : str, optional
    dark : bool, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    factor, suffix = _axis_scale(impacts, unit)
    scaled = np.asarray(impacts, dtype=float) / factor
    tok = tokens(dark)

    figure.add_trace(
        go.Bar(
            x=[f"{int(rp)}y" for rp in return_periods],
            y=scaled,
            marker={"color": categorical(dark)[0], "cornerradius": _CORNER_RADIUS},
            text=[f"{value:,.3g}" for value in scaled],
            textposition="outside",
            textfont={"color": tok["text_secondary"]},
            hovertemplate=(
                f"Return period %{{x}}<br>Impact %{{y:,.3g}} {suffix}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(bargap=0.35)
    figure.update_xaxes(title_text="Return period", showgrid=False)
    figure.update_yaxes(title_text=f"Impact ({suffix})".strip(), rangemode="tozero")
    return figure


def series_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    unit: str = "",
    dark: bool = False,
    title: str = "",
    x_title: str = "",
    horizontal: bool = False,
) -> go.Figure:
    """A single-series bar chart with the UI's mark specs.

    Parameters
    ----------
    labels : sequence of str
    values : sequence of float
    unit : str, optional
    dark : bool, optional
    title : str, optional
    x_title : str, optional
        Label for the categorical axis.
    horizontal : bool, optional
        Lay the bars out horizontally, for long category names. Default: False.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    factor, suffix = _axis_scale(values, unit)
    scaled = np.asarray(values, dtype=float) / factor
    value_title = f"Impact ({suffix})".strip() if unit else "Value"

    if horizontal:
        figure.add_trace(
            go.Bar(
                y=list(labels),
                x=scaled,
                orientation="h",
                marker={"color": categorical(dark)[0], "cornerradius": _CORNER_RADIUS},
                hovertemplate=f"%{{y}}<br>%{{x:,.3g}} {suffix}<extra></extra>",
                showlegend=False,
            )
        )
        figure.update_xaxes(title_text=value_title, rangemode="tozero")
        figure.update_yaxes(title_text=x_title, autorange="reversed", showgrid=False)
    else:
        figure.add_trace(
            go.Bar(
                x=list(labels),
                y=scaled,
                marker={"color": categorical(dark)[0], "cornerradius": _CORNER_RADIUS},
                hovertemplate=f"%{{x}}<br>%{{y:,.3g}} {suffix}<extra></extra>",
                showlegend=False,
            )
        )
        figure.update_xaxes(title_text=x_title, showgrid=False)
        figure.update_yaxes(title_text=value_title, rangemode="tozero")

    figure.update_layout(bargap=0.3)
    return figure


def impact_function_chart(
    curves: Dict[str, Dict[str, Sequence[float]]],
    intensity_unit: str = "",
    dark: bool = False,
    title: str = "Vulnerability curves",
) -> go.Figure:
    """Mean damage ratio against hazard intensity, one line per function.

    Parameters
    ----------
    curves : dict
        Mapping of label to ``{'intensity': [...], 'mdr': [...]}``.
    intensity_unit : str, optional
        Unit of the hazard intensity, used on the x-axis.
    dark : bool, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    palette = categorical(dark)

    for index, (label, curve) in enumerate(curves.items()):
        figure.add_trace(
            go.Scatter(
                x=np.asarray(curve["intensity"], dtype=float),
                y=np.asarray(curve["mdr"], dtype=float) * 100.0,
                name=label,
                mode="lines",
                line={
                    "width": _LINE_WIDTH,
                    "color": palette[index % MAX_CATEGORICAL_SERIES],
                },
                hovertemplate=(
                    f"<b>{label}</b><br>Intensity %{{x:,.3g}} {intensity_unit}"
                    "<br>Damage %{y:,.1f}%<extra></extra>"
                ),
            )
        )

    figure.update_layout(hovermode="x unified", showlegend=len(curves) > 1)
    figure.update_xaxes(
        title_text=f"Hazard intensity ({intensity_unit})".strip(), rangemode="tozero"
    )
    figure.update_yaxes(title_text="Mean damage ratio (%)", rangemode="tozero")
    return figure


def point_map(
    frame: pd.DataFrame,
    value_column: str,
    unit: str = "",
    dark: bool = False,
    title: str = "",
    basemap: bool = False,
) -> go.Figure:
    """Geographic scatter of exposure points coloured by magnitude.

    Magnitude is a sequential encoding: one hue, light to dark. Without a
    basemap the figure is a plain equal-aspect lon/lat plot, which renders with
    no network access; with one it uses OpenStreetMap tiles fetched by the
    browser.

    Parameters
    ----------
    frame : pandas.DataFrame
        Must contain ``latitude``, ``longitude`` and ``value_column``.
    value_column : str
        Column carrying the magnitude to colour by.
    unit : str, optional
    dark : bool, optional
    title : str, optional
    basemap : bool, optional
        Draw OpenStreetMap tiles behind the points. Requires the viewer's
        browser to reach the tile server. Default: False.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title, height=460)
    if frame.empty:
        figure.add_annotation(
            text="No points to show", showarrow=False, font={"size": 14}
        )
        return figure

    factor, suffix = _axis_scale(frame[value_column].values, unit)
    scaled = frame[value_column].to_numpy(dtype=float) / factor
    colorbar = {
        "title": {"text": suffix or value_column, "side": "right"},
        "thickness": 12,
        "outlinewidth": 0,
        "tickfont": {"color": tokens(dark)["text_muted"]},
    }
    marker = {
        "size": 8,
        "color": scaled,
        "colorscale": [
            [index / (len(sequential_scale(dark)) - 1), color]
            for index, color in enumerate(sequential_scale(dark))
        ],
        "colorbar": colorbar,
        "opacity": 0.85,
    }
    hover = (
        "%{customdata[0]:.3f}, %{customdata[1]:.3f}"
        f"<br>%{{marker.color:,.3g}} {suffix}<extra></extra>"
    )
    custom = np.column_stack(
        [frame["latitude"].to_numpy(float), frame["longitude"].to_numpy(float)]
    )

    if basemap:
        figure.add_trace(
            go.Scattermap(
                lat=frame["latitude"],
                lon=frame["longitude"],
                mode="markers",
                marker=marker,
                customdata=custom,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        figure.update_layout(
            map={
                "style": "carto-darkmatter" if dark else "carto-positron",
                "center": {
                    "lat": float(frame["latitude"].mean()),
                    "lon": float(frame["longitude"].mean()),
                },
                "zoom": 4,
            },
            margin={"l": 0, "r": 0, "t": 48, "b": 0},
        )
        return figure

    marker["line"] = {"width": 0}
    figure.add_trace(
        go.Scattergl(
            x=frame["longitude"],
            y=frame["latitude"],
            mode="markers",
            marker=marker,
            customdata=custom,
            hovertemplate=hover,
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text="Longitude", showgrid=True)
    figure.update_yaxes(
        title_text="Latitude",
        showgrid=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return figure


def cost_benefit_chart(
    measures: Sequence[str],
    benefits: Sequence[float],
    bc_ratios: Sequence[float],
    total_climate_risk: float,
    unit: str = "",
    horizon_years: Optional[int] = None,
    dark: bool = False,
    title: str = "Adaptation cost-benefit",
) -> go.Figure:
    """CLIMADA's cost-benefit chart: bar width is benefit, height is B/C ratio.

    Each measure is a rectangle whose area equals its cost. Measures are laid
    out left to right from the most to the least efficient, so the horizontal
    extent of the whole block is the risk that adaptation removes, and anything
    to the right of it up to the total climate risk stays as residual risk.

    Parameters
    ----------
    measures : sequence of str
        Measure names.
    benefits : sequence of float
        Net present value of averted damage per measure.
    bc_ratios : sequence of float
        Benefit/cost ratio per measure.
    total_climate_risk : float
        Net present value of the unmitigated risk over the horizon.
    unit : str, optional
    horizon_years : int, optional
        Length of the appraisal horizon, named on the x-axis label.
    dark : bool, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title, height=460)
    palette = categorical(dark)
    tok = tokens(dark)

    benefits = np.asarray(benefits, dtype=float)
    bc_ratios = np.asarray(bc_ratios, dtype=float)
    factor, suffix = _axis_scale(np.append(benefits, total_climate_risk), unit)

    order = np.argsort(-np.nan_to_num(bc_ratios, nan=-np.inf))
    cursor = 0.0
    for slot, index in enumerate(order):
        width = float(benefits[index]) / factor
        height = float(bc_ratios[index])
        if not np.isfinite(width) or width <= 0 or not np.isfinite(height):
            continue
        # A 2px surface gap between neighbouring fills, expressed in data units.
        gap = min(width * 0.02, 0.01 * max(cursor + width, 1.0))
        figure.add_trace(
            go.Bar(
                x=[cursor + width / 2],
                y=[height],
                width=[max(width - gap, width * 0.9)],
                name=measures[index],
                marker={
                    "color": palette[slot % MAX_CATEGORICAL_SERIES],
                    "cornerradius": _CORNER_RADIUS,
                },
                customdata=[[width]],
                hovertemplate=(
                    f"<b>{measures[index]}</b>"
                    f"<br>Averted damage %{{customdata[0]:,.3g}} {suffix}"
                    f"<br>Benefit/cost %{{y:,.2f}}<extra></extra>"
                ),
            )
        )
        cursor += width

    risk_scaled = float(total_climate_risk) / factor
    if np.isfinite(risk_scaled) and risk_scaled > 0:
        figure.add_vline(
            x=risk_scaled,
            line={"color": tok["text_muted"], "width": 1, "dash": "dot"},
            annotation={
                "text": f"Total climate risk {risk_scaled:,.3g} {suffix}",
                "font": {"color": tok["text_secondary"], "size": 12},
                "yanchor": "bottom",
            },
        )
    figure.add_hline(
        y=1.0,
        line={"color": tok["axis"], "width": 1},
        annotation={
            "text": "break-even",
            "font": {"color": tok["text_muted"], "size": 11},
            "yanchor": "bottom",
            "xanchor": "left",
        },
        annotation_position="top left",
    )

    horizon = f" over {horizon_years} years" if horizon_years else ""
    figure.update_layout(barmode="overlay", showlegend=len(measures) > 1)
    figure.update_xaxes(
        title_text=f"NPV averted damage{horizon} ({suffix})".strip(),
        rangemode="tozero",
        showgrid=False,
    )
    figure.update_yaxes(title_text="Benefit/cost ratio", rangemode="tozero")
    return figure


def _bcr_label(value: float) -> str:
    """Direct label for a benefit/cost bar, naming the verdict in words.

    Colour alone never carries the pays-for-itself reading; the text does.

    Parameters
    ----------
    value : float
        Benefit/cost ratio.

    Returns
    -------
    str
    """
    if not np.isfinite(value):
        # An unbounded ratio means no cost was entered, not a perfect measure.
        return "  no cost entered" if value > 0 else "  not computed"
    return f"{value:,.2f}" + ("  pays for itself" if value >= 1 else "  below cost")


def _bcr_color(value: float) -> str:
    """Status colour for a benefit/cost bar.

    A ratio that is not a number is a warning about the inputs, not a verdict
    on the measure, so it never takes the good or critical colour.

    Parameters
    ----------
    value : float
        Benefit/cost ratio.

    Returns
    -------
    str
    """
    if not np.isfinite(value):
        return STATUS["warning"]
    return STATUS["good"] if value >= 1 else STATUS["critical"]


def measure_bcr_bars(
    measures: Sequence[str],
    bc_ratios: Sequence[float],
    dark: bool = False,
    title: str = "Benefit/cost ratio by measure",
) -> go.Figure:
    """Benefit/cost ratios ranked, with break-even marked.

    Measures at or above 1.0 pay for themselves; the status colour and the
    ``break-even`` line together carry that reading, never colour alone.

    Parameters
    ----------
    measures : sequence of str
    bc_ratios : sequence of float
    dark : bool, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    tok = tokens(dark)
    ratios = np.asarray(bc_ratios, dtype=float)
    order = np.argsort(-np.nan_to_num(ratios, nan=-np.inf))
    names = [measures[index] for index in order]
    values = ratios[order]

    figure.add_trace(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker={
                "color": [_bcr_color(value) for value in values],
                "cornerradius": _CORNER_RADIUS,
            },
            text=[_bcr_label(value) for value in values],
            textposition="outside",
            textfont={"color": tok["text_secondary"]},
            hovertemplate="%{y}<br>Benefit/cost %{x:,.2f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_vline(x=1.0, line={"color": tok["axis"], "width": 1, "dash": "dash"})
    figure.update_layout(bargap=0.35)
    figure.update_xaxes(title_text="Benefit/cost ratio", rangemode="tozero")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return figure


def residual_risk_bars(
    measures: Sequence[str],
    averted: Sequence[float],
    residual: Sequence[float],
    unit: str = "",
    dark: bool = False,
    title: str = "Annual risk: averted and residual",
) -> go.Figure:
    """Stacked split of annual risk into the part averted and the part left.

    Parameters
    ----------
    measures : sequence of str
    averted : sequence of float
        Annual risk removed by each measure.
    residual : sequence of float
        Annual risk remaining under each measure.
    unit : str, optional
    dark : bool, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    palette = categorical(dark)
    factor, suffix = _axis_scale(np.concatenate([averted, residual]), unit)

    figure.add_trace(
        go.Bar(
            x=list(measures),
            y=np.asarray(averted, dtype=float) / factor,
            name="Averted",
            marker={"color": palette[0], "cornerradius": _CORNER_RADIUS},
            hovertemplate=f"%{{x}}<br>Averted %{{y:,.3g}} {suffix}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=list(measures),
            y=np.asarray(residual, dtype=float) / factor,
            name="Residual",
            marker={"color": palette[1], "cornerradius": _CORNER_RADIUS},
            hovertemplate=f"%{{x}}<br>Residual %{{y:,.3g}} {suffix}<extra></extra>",
        )
    )
    figure.update_layout(barmode="stack", bargap=0.35)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(
        title_text=f"Annual risk ({suffix})".strip(), rangemode="tozero"
    )
    return figure


def waterfall_chart(
    present_year: int,
    future_year: int,
    present_risk: float,
    development_delta: float,
    climate_delta: float,
    unit: str = "",
    dark: bool = False,
    title: str = "What drives the growth in risk",
) -> go.Figure:
    """Risk today, the two drivers of its growth, and risk at the horizon.

    Parameters
    ----------
    present_year : int
    future_year : int
    present_risk : float
        Annual risk today.
    development_delta : float
        Risk added by socio-economic development.
    climate_delta : float
        Risk added by the change in hazard.
    unit : str, optional
    dark : bool, optional
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, title)
    tok = tokens(dark)
    future_risk = present_risk + development_delta + climate_delta
    factor, suffix = _axis_scale(
        [present_risk, future_risk, abs(development_delta), abs(climate_delta)], unit
    )

    figure.add_trace(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=[
                f"Risk {present_year}",
                "Economic development",
                "Climate change",
                f"Risk {future_year}",
            ],
            y=[
                present_risk / factor,
                development_delta / factor,
                climate_delta / factor,
                future_risk / factor,
            ],
            text=[
                f"{present_risk / factor:,.3g}",
                f"{development_delta / factor:+,.3g}",
                f"{climate_delta / factor:+,.3g}",
                f"{future_risk / factor:,.3g}",
            ],
            textposition="outside",
            textfont={"color": tok["text_secondary"]},
            connector={"line": {"color": tok["axis"], "width": 1}},
            increasing={"marker": {"color": STATUS["critical"]}},
            decreasing={"marker": {"color": STATUS["good"]}},
            totals={"marker": {"color": categorical(dark)[0]}},
            hovertemplate=f"%{{x}}<br>%{{y:,.3g}} {suffix}<extra></extra>",
        )
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(
        title_text=f"Average annual impact ({suffix})".strip(), rangemode="tozero"
    )
    return figure


def discount_rate_chart(
    years: Sequence[int], rates: Sequence[float], dark: bool = False
) -> go.Figure:
    """The discount rate path over the appraisal horizon.

    Parameters
    ----------
    years : sequence of int
    rates : sequence of float
        Rates as fractions, e.g. 0.02 for 2%.
    dark : bool, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = _base_figure(dark, "Discount rate", height=240)
    figure.add_trace(
        go.Scatter(
            x=list(years),
            y=np.asarray(rates, dtype=float) * 100.0,
            mode="lines",
            line={"width": _LINE_WIDTH, "color": categorical(dark)[0]},
            hovertemplate="%{x}<br>%{y:,.2f}%<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text="Year", showgrid=False)
    figure.update_yaxes(title_text="Rate (%)", rangemode="tozero")
    return figure
