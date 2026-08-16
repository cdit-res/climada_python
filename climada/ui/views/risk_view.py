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

Risk page: the physical climate risk assessment, before any adaptation.
"""

# The view layer is the boundary between CLIMADA and the browser: any
# failure below must become a message on the page, never a crashed session.
# pylint: disable=broad-exception-caught


import numpy as np
import pandas as pd
import streamlit as st

from climada.ui import analysis, charts, components, heat, state
from climada.ui.formatting import fmt_compact, fmt_percent, fmt_value

DEFAULT_RP_CHOICES = [2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]


def render() -> None:
    """Draw the physical risk page."""
    components.page_header(
        "Physical risk",
        "How much damage the hazard does to the exposures today, expressed as "
        "an annual average and as losses at chosen return periods.",
    )
    if not components.requires_inputs():
        return

    _controls()
    risk = state.get("risk")
    if risk is None:
        st.info("Press **Run risk assessment** to compute the impact.")
        return

    _headline(risk)
    st.divider()
    _curves(risk)
    st.divider()
    _geography(risk)
    st.divider()
    _events(risk)


def _controls() -> None:
    """Return-period selection and the run button."""
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            return_periods = st.multiselect(
                "Return periods to report (years)",
                DEFAULT_RP_CHOICES,
                default=[
                    rp
                    for rp in analysis.DEFAULT_RETURN_PERIODS
                    if rp in DEFAULT_RP_CHOICES
                ],
                key="w_risk_rps",
            )
            save_mat = st.checkbox(
                "Keep the full event-by-asset matrix",
                value=False,
                help=(
                    "Needed only for per-event asset detail. Uses a lot of "
                    "memory on large exposure sets."
                ),
                key="w_risk_savemat",
            )
        with right:
            st.write("")
            run = st.button("Run risk assessment", type="primary", key="w_risk_run")

    if not run:
        return

    if not return_periods:
        st.warning("Pick at least one return period.")
        return

    try:
        with st.spinner("Computing impact..."):
            risk = analysis.compute_risk(
                state.get("exposures"),
                state.get("impf_set"),
                state.get("hazard"),
                return_periods=sorted(return_periods),
                save_mat=save_mat,
            )
    except Exception as err:
        components.error_box(err, "compute the impact")
        return

    state.put("risk", risk)
    st.rerun()


def _headline(risk: analysis.RiskResult) -> None:
    """Stat tiles summarising the assessment."""
    metric = _heat_metric(risk)
    if metric is not None:
        _heat_headline(risk, metric)
        return

    components.stat_row(
        [
            (
                "Average annual impact",
                fmt_compact(risk.aai, risk.unit),
                "Expected loss per year, averaged over the event set",
            ),
            (
                "Annual loss ratio",
                fmt_percent(risk.loss_ratio, digits=3),
                "Average annual impact as a share of the total exposed value",
            ),
            (
                "100-year loss",
                fmt_compact(risk.rp_value(100), risk.unit),
                "Loss exceeded on average once a century",
            ),
            (
                "Largest single event",
                fmt_compact(risk.max_event_impact, risk.unit),
                "Worst event in the hazard set",
            ),
        ]
    )
    st.caption(
        f"Total exposed value {fmt_value(risk.total_value, risk.unit)} - "
        f"hazard '{risk.haz_type}' with {risk.impact.at_event.size:,} events."
    )


def _impact_unit(risk: analysis.RiskResult) -> str:
    """The unit the impact numbers are in, which is not the exposure unit.

    Parameters
    ----------
    risk : RiskResult

    Returns
    -------
    str
    """
    return heat.impact_unit(risk.haz_type, state.get("heat_metric"), risk.unit)


def _heat_metric(risk: analysis.RiskResult):
    """The heat metric in play, or None when this is not a heat analysis."""
    return heat.metric_for(risk.haz_type, state.get("heat_metric"))


def _heat_headline(risk: analysis.RiskResult, metric) -> None:
    """Stat tiles that say deaths and people rather than loss and value."""
    tiles = [
        (
            f"Annual {metric.unit}",
            fmt_compact(risk.aai, ""),
            f"Expected {metric.annual_noun}, averaged over the record",
        )
    ]
    if metric.per_capita_basis and risk.total_value:
        rate = risk.aai / risk.total_value * metric.per_capita_basis
        tiles.append(
            (
                f"Per {metric.per_capita_basis:,.0f} people",
                f"{rate:,.1f}",
                f"Annual {metric.unit} per {metric.per_capita_basis:,.0f} of the "
                "exposed population",
            )
        )
    else:
        tiles.append(
            (
                "Exposed population",
                fmt_compact(risk.total_value, metric.exposure_unit),
                "Total exposure the metric is computed over",
            )
        )
    tiles.append(
        (
            "Worst day in the record",
            fmt_compact(risk.max_event_impact, ""),
            f"{metric.unit.capitalize()} on the single worst day",
        )
    )

    summary = heat.hazard_temperature_summary(state.get("hazard"))
    tiles.append(
        (
            "Peak temperature",
            f"{summary['peak']:,.1f} {state.get('hazard').units}",
            "Hottest value anywhere in the record",
        )
    )
    components.stat_row(tiles)

    st.caption(
        f"{metric.label}: {metric.description} Computed over "
        f"{risk.impact.at_event.size:,} daily fields spanning "
        f"{summary['years']:,.1f} years, against "
        f"{fmt_value(risk.total_value, metric.exposure_unit)}."
    )


def _curves(risk: analysis.RiskResult) -> None:
    """Exceedance curve and return-period bars, each with its table."""
    st.subheader("Loss exceedance")
    left, right = st.columns(2)

    with left:
        unit = _impact_unit(risk)
        figure = charts.exceedance_curve(
            {"Current risk": (risk.freq_curve_return_per, risk.freq_curve_impact)},
            unit=unit,
            dark=state.dark_mode(),
        )
        curve_frame = pd.DataFrame(
            {
                "Return period (years)": risk.freq_curve_return_per,
                f"Impact ({unit})": risk.freq_curve_impact,
            }
        )
        components.chart_with_table(
            figure,
            curve_frame,
            table_label="Show the full curve",
            download_name="exceedance_curve.csv",
            key="risk-efc",
        )

    with right:
        figure = charts.return_period_bars(
            risk.return_periods,
            risk.rp_impact,
            unit=_impact_unit(risk),
            dark=state.dark_mode(),
        )
        components.chart_with_table(
            figure,
            analysis.exceedance_table(risk),
            table_label="Show the return-period table",
            download_name="return_periods.csv",
            key="risk-rp",
        )

    if _heat_metric(risk) is not None:
        st.caption(
            "Each event is one day, so the curve reads: a day this bad or "
            "worse comes round once every N years. Return periods longer than "
            "the record are extrapolation."
        )
    else:
        st.caption(
            "The curve reads: a loss of this size or larger happens on average "
            "once every N years. It is interpolated from the event set, so "
            "return periods beyond the longest event interval are extrapolation."
        )


def _geography(risk: analysis.RiskResult) -> None:
    """Where the expected annual impact lands."""
    st.subheader("Where the risk sits")

    limit = int(state.get("map_point_limit"))
    points = analysis.impact_point_table(risk.impact, limit=limit)
    total_points = np.asarray(risk.impact.eai_exp).size
    if total_points > limit:
        st.caption(f"Showing the {limit:,} highest-impact of {total_points:,} points.")

    unit = _impact_unit(risk)
    figure = charts.point_map(
        points,
        "eai",
        unit=unit,
        dark=state.dark_mode(),
        title="Expected annual impact per location",
        basemap=bool(state.get("basemap")),
    )
    components.chart_with_table(
        figure,
        points.head(200).rename(columns={"eai": f"Expected annual impact ({unit})"}),
        table_label="Show the worst-affected locations",
        download_name="impact_by_location.csv",
        key="risk-map",
    )

    regions = analysis.impact_by_region(risk.impact, state.get("exposures"))
    if regions is not None and not regions.empty:
        st.markdown("**By region**")
        figure = charts.series_bar_chart(
            [str(value) for value in regions["Region"]],
            regions["Expected annual impact"].values,
            unit=_impact_unit(risk),
            dark=state.dark_mode(),
            title="Expected annual impact by region",
            x_title="Region id",
            horizontal=len(regions) > 6,
        )
        components.chart_with_table(
            figure,
            regions,
            table_label="Show the regional table",
            download_name="impact_by_region.csv",
            key="risk-region",
        )


def _events(risk: analysis.RiskResult) -> None:
    """Worst events and, where dates allow, the annual history."""
    st.subheader("Events")
    events = analysis.top_events_table(risk.impact, limit=15)
    if events.empty:
        st.info("The hazard set has no events to rank.")
        return

    impact_column = [col for col in events.columns if col.startswith("Impact")][0]
    figure = charts.series_bar_chart(
        list(events["Event"].astype(str)),
        events[impact_column].values,
        unit=_impact_unit(risk),
        dark=state.dark_mode(),
        title="Most damaging events",
        x_title="Event",
        horizontal=True,
    )
    components.chart_with_table(
        figure,
        events,
        table_label="Show the event table",
        download_name="top_events.csv",
        key="risk-events",
    )

    annual = analysis.annual_impact_table(risk.impact)
    if annual is not None and len(annual) > 1:
        value_column = [col for col in annual.columns if col != "Year"][0]
        figure = charts.series_bar_chart(
            [str(year) for year in annual["Year"]],
            annual[value_column].values,
            unit=_impact_unit(risk),
            dark=state.dark_mode(),
            title="Impact per year in the event set",
            x_title="Year",
        )
        components.chart_with_table(
            figure,
            annual,
            table_label="Show the annual table",
            download_name="impact_per_year.csv",
            key="risk-annual",
        )
