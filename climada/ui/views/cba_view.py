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

Cost-benefit page: rank adaptation measures over an appraisal horizon.
"""

# The view layer is the boundary between CLIMADA and the browser: any
# failure below must become a message on the page, never a crashed session.
# pylint: disable=broad-exception-caught


from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from climada.ui import analysis, charts, components, state
from climada.ui.formatting import fmt_compact, fmt_ratio


def render() -> None:
    """Draw the cost-benefit page."""
    components.page_header(
        "Cost-benefit analysis",
        "Discount the damage each measure averts over the appraisal horizon and "
        "set it against what the measure costs.",
    )
    if not components.requires_inputs(needs_measures=True):
        return

    _settings()
    cost_ben = state.get("cost_benefit")
    if cost_ben is None:
        st.info("Set the horizon and scenario above, then press **Run cost-benefit**.")
        return

    _headline(cost_ben)
    st.divider()
    _ranking(cost_ben)
    st.divider()
    _risk_drivers()
    st.divider()
    _residual(cost_ben)
    st.divider()
    _combine(cost_ben)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def _settings() -> None:
    """Horizon, discounting, scenario and the run button."""
    with st.container(border=True):
        horizon, discounting = st.columns(2)

        with horizon:
            st.markdown("**Appraisal horizon**")
            present = st.number_input(
                "Present year",
                min_value=1900,
                max_value=2200,
                value=int(state.get("present_year")),
                step=1,
                key="w_cba_present",
            )
            future = st.number_input(
                "Horizon year",
                min_value=1901,
                max_value=2200,
                value=int(max(state.get("future_year"), present + 1)),
                step=1,
                key="w_cba_future",
            )
            if future <= present:
                st.error("The horizon year must be later than the present year.")

        with discounting:
            st.markdown("**Discounting**")
            rate = st.slider(
                "Annual discount rate",
                min_value=0.0,
                max_value=0.12,
                value=float(state.get("disc_rate")),
                step=0.005,
                format="%.3f",
                help="Applied to the stream of averted damage. Measure costs are "
                "entered already discounted.",
                key="w_cba_rate",
            )
            metric = st.selectbox(
                "Risk metric benefits are measured in",
                list(analysis.RISK_FUNCTIONS),
                index=list(analysis.RISK_FUNCTIONS).index(state.get("risk_metric")),
                key="w_cba_metric",
            )
            time_dep = st.slider(
                "Impact growth path",
                min_value=0.25,
                max_value=4.0,
                value=float(state.get("imp_time_depen")),
                step=0.25,
                help="1 is a straight line from present to horizon risk. Below 1 "
                "front-loads the growth, above 1 back-loads it.",
                key="w_cba_timedep",
            )

        st.markdown("**Scenario at the horizon**")
        use_future = st.checkbox(
            "Model a changing world between the two years",
            value=bool(state.get("use_future")),
            help="Off means today's hazard and today's assets throughout, and "
            "benefits accrue at a constant annual rate.",
            key="w_cba_usefuture",
        )
        growth = intensity = frequency = 1.0
        if use_future:
            left, middle, right = st.columns(3)
            growth = left.number_input(
                "Exposure growth over the horizon",
                min_value=0.1,
                max_value=10.0,
                value=float(state.get("growth_factor")),
                step=0.05,
                help="1.4 means asset values are 40% higher at the horizon year.",
                key="w_cba_growth",
            )
            intensity = middle.number_input(
                "Hazard intensity factor",
                min_value=0.5,
                max_value=3.0,
                value=float(state.get("intensity_factor")),
                step=0.01,
                help="Scales every event's intensity. A crude stand-in for a "
                "downscaled future hazard set.",
                key="w_cba_intensity",
            )
            frequency = right.number_input(
                "Hazard frequency factor",
                min_value=0.1,
                max_value=5.0,
                value=float(state.get("frequency_factor")),
                step=0.05,
                help="Scales how often every event occurs.",
                key="w_cba_frequency",
            )
            if intensity != 1.0 or frequency != 1.0:
                st.caption(
                    "Uniform scaling factors are a screening device, not a climate "
                    "projection. For a defensible study load a future hazard set "
                    "from the Data API on the **Data** page."
                )

        run = st.button(
            "Run cost-benefit",
            type="primary",
            disabled=future <= present,
            key="w_cba_run",
        )

    state.put("present_year", int(present))
    state.put("future_year", int(future))
    state.put("disc_rate", float(rate))
    state.put("risk_metric", metric)
    state.put("imp_time_depen", float(time_dep))
    state.put("use_future", bool(use_future))
    state.put("growth_factor", float(growth))
    state.put("intensity_factor", float(intensity))
    state.put("frequency_factor", float(frequency))

    if run:
        _run()


def _build_scenario(
    hazard, exposures, impf_set, measure_set, disc_rates, future_year: int
) -> Tuple[Optional[Any], Optional[Any]]:
    """The hazard and entity describing the horizon year, per the settings.

    Returns ``(None, None)`` when the user asked for a static world, which is
    how CLIMADA is told to hold hazard and assets fixed.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Present-day hazard.
    exposures : climada.entity.Exposures
        Present-day exposures.
    impf_set : climada.entity.ImpactFuncSet
    measure_set : climada.entity.MeasureSet
    disc_rates : climada.entity.DiscRates
    future_year : int
        Horizon year stamped on the future exposures.

    Returns
    -------
    tuple
        ``(haz_future, ent_future)``, either of which may be ``None``.
    """
    if not state.get("use_future"):
        return None, None

    intensity = float(state.get("intensity_factor"))
    frequency = float(state.get("frequency_factor"))
    loaded_future = state.get("future_hazard")

    haz_future: Optional[Any] = None
    if loaded_future is not None:
        haz_future = loaded_future
    elif intensity != 1.0 or frequency != 1.0:
        haz_future = analysis.scale_hazard(hazard, intensity, frequency)

    ent_future = analysis.build_entity(
        analysis.grow_exposures(
            exposures, float(state.get("growth_factor")), ref_year=future_year
        ),
        impf_set,
        measure_set,
        disc_rates,
        ref_year=future_year,
    )
    return haz_future, ent_future


def _run() -> None:
    """Assemble entities and hazards from the settings, then run CLIMADA."""
    present = int(state.get("present_year"))
    future = int(state.get("future_year"))
    hazard = state.get("hazard")
    exposures = state.get("exposures")
    impf_set = state.get("impf_set")
    haz_type = hazard.haz_type

    try:
        measure_set = analysis.build_measure_set(state.measure_rows(), haz_type)
    except ValueError as err:
        st.error(str(err))
        return

    disc_rates = analysis.build_disc_rates(
        float(state.get("disc_rate")), present, future
    )

    try:
        entity = analysis.build_entity(
            exposures, impf_set, measure_set, disc_rates, ref_year=present
        )
    except Exception as err:
        components.error_box(err, "assemble the present-day entity")
        return

    haz_future, ent_future = _build_scenario(
        hazard, exposures, impf_set, measure_set, disc_rates, future
    )

    try:
        with st.spinner("Running the cost-benefit calculation..."):
            cost_ben = analysis.run_cost_benefit(
                hazard,
                entity,
                haz_future=haz_future,
                ent_future=ent_future,
                future_year=future,
                risk_func=analysis.RISK_FUNCTIONS[state.get("risk_metric")],
                imp_time_depen=float(state.get("imp_time_depen")),
                save_imp=True,
            )
    except Exception as err:
        components.error_box(err, "run the cost-benefit calculation")
        return

    waterfall = None
    if ent_future is not None:
        try:
            waterfall = analysis.waterfall_components(
                hazard,
                entity,
                haz_future if haz_future is not None else hazard,
                ent_future,
                risk_func=analysis.RISK_FUNCTIONS[state.get("risk_metric")],
            )
        except Exception:  # the waterfall is a nice-to-have
            waterfall = None

    state.put("cost_benefit", cost_ben)
    state.put("waterfall", waterfall)
    state.put("combined", None)
    state.put(
        "cost_benefit_meta",
        {
            "present_year": present,
            "future_year": future,
            "disc_rate": float(state.get("disc_rate")),
            "risk_metric": state.get("risk_metric"),
            "imp_time_depen": float(state.get("imp_time_depen")),
            "growth_factor": float(state.get("growth_factor")),
            "intensity_factor": float(state.get("intensity_factor")),
            "frequency_factor": float(state.get("frequency_factor")),
            "scenario": bool(state.get("use_future")),
        },
    )
    st.rerun()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def _headline(cost_ben) -> None:
    """Portfolio-level stat tiles."""
    summary = analysis.cost_benefit_summary(cost_ben)
    unit = cost_ben.unit
    years = summary["future_year"] - summary["present_year"] + 1

    components.stat_row(
        [
            (
                f"Total climate risk ({years}y NPV)",
                fmt_compact(summary["total_climate_risk"], unit),
                "Discounted damage over the horizon if nothing is done",
            ),
            (
                "Averted by all measures",
                fmt_compact(summary["averted"], unit),
                "Sum of the discounted benefits",
            ),
            (
                "Residual risk",
                fmt_compact(summary["residual"], unit),
                "What is left after every measure is implemented",
            ),
            (
                "Portfolio benefit/cost",
                fmt_ratio(summary["portfolio_bcr"]),
                "Total averted damage divided by total cost",
            ),
        ]
    )

    meta = state.get("cost_benefit_meta") or {}
    line = (
        f"{summary['present_year']}-{summary['future_year']} at "
        f"{meta.get('disc_rate', 0):.1%} discount, benefits measured as "
        f"{meta.get('risk_metric', 'average annual impact').lower()}."
    )
    if np.isfinite(summary["annual_risk_present"]):
        line += (
            f" Annual risk grows from "
            f"{fmt_compact(summary['annual_risk_present'], unit)} to "
            f"{fmt_compact(summary['annual_risk_future'], unit)}."
        )
    else:
        line += f" Annual risk {fmt_compact(summary['annual_risk_future'], unit)}."
    st.caption(line)

    if summary["residual"] < 0:
        st.warning(
            "The measures together avert more than the total climate risk. That "
            "happens when their effects overlap and are counted separately -- use "
            "**Combine measures** below for the joint effect."
        )


def _ranking(cost_ben) -> None:
    """The cost-benefit chart and the ranked table."""
    st.subheader("Which measures pay for themselves")
    table = analysis.cost_benefit_table(cost_ben)
    if table.empty:
        st.info("No measure produced a benefit to rank.")
        return

    unit = cost_ben.unit
    benefit_column = f"Benefit ({unit})"
    summary = analysis.cost_benefit_summary(cost_ben)
    years = summary["future_year"] - summary["present_year"] + 1

    figure = charts.cost_benefit_chart(
        list(table["Measure"]),
        table[benefit_column].values,
        table["Benefit/cost ratio"].values,
        total_climate_risk=summary["total_climate_risk"],
        unit=unit,
        horizon_years=years,
        dark=state.dark_mode(),
    )
    st.plotly_chart(figure, use_container_width=True, key="cba-main")
    st.caption(
        "Each block's width is the damage that measure averts and its height is "
        "the benefit per unit of cost, so the block's area is what it costs. "
        "Anything below the break-even line costs more than it saves."
    )

    left, right = st.columns([1, 1])
    with left:
        components.chart_with_table(
            charts.measure_bcr_bars(
                list(table["Measure"]),
                table["Benefit/cost ratio"].values,
                dark=state.dark_mode(),
            ),
            key="cba-bcr",
        )
    with right:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                f"Cost ({unit})": st.column_config.NumberColumn(format="compact"),
                benefit_column: st.column_config.NumberColumn(format="compact"),
                f"Net benefit ({unit})": st.column_config.NumberColumn(
                    format="compact"
                ),
                "Benefit/cost ratio": st.column_config.NumberColumn(format="%.2f"),
                "Residual risk share": st.column_config.NumberColumn(format="percent"),
            },
        )
        components.download_frame(table, "cost_benefit.csv", key="cba-table-dl")


def _risk_drivers() -> None:
    """The waterfall, when a future scenario was modelled."""
    waterfall = state.get("waterfall")
    if waterfall is None:
        return

    st.subheader("What drives the growth in risk")
    figure = charts.waterfall_chart(
        waterfall.present_year,
        waterfall.future_year,
        waterfall.present_risk,
        waterfall.development_delta,
        waterfall.climate_delta,
        unit=waterfall.unit,
        dark=state.dark_mode(),
    )
    components.chart_with_table(
        figure,
        waterfall.to_frame(),
        table_label="Show the components",
        download_name="risk_drivers.csv",
        key="cba-waterfall",
    )
    st.caption(
        "Economic development is the extra risk from having more assets exposed; "
        "climate change is the extra risk from the hazard itself changing. They "
        "are computed in that order, so the split is not symmetric."
    )


def _residual(cost_ben) -> None:
    """Residual annual risk under each measure, and the exceedance curves."""
    st.subheader("Residual risk under each measure")
    risk_table = analysis.measure_risk_table(cost_ben)
    measures = risk_table[risk_table["Measure"] != analysis.NO_MEASURE]
    if measures.empty:
        return

    unit = cost_ben.unit
    residual_column = f"Residual annual risk ({unit})"
    averted_column = f"Annual risk averted ({unit})"

    figure = charts.residual_risk_bars(
        list(measures["Measure"]),
        measures[averted_column].values,
        measures[residual_column].values,
        unit=unit,
        dark=state.dark_mode(),
    )
    components.chart_with_table(
        figure,
        risk_table,
        table_label="Show the residual risk table",
        download_name="residual_risk.csv",
        key="cba-residual",
    )

    curves = {}
    for name, values in cost_ben.imp_meas_future.items():
        efc = values.get("efc")
        if efc is None:
            continue
        label = "No measure" if name == analysis.NO_MEASURE else name
        curves[label] = (efc.return_per, efc.impact)
        if len(curves) >= charts.MAX_CATEGORICAL_SERIES:
            break

    if len(curves) > 1:
        st.markdown("**Loss exceedance with and without each measure**")
        figure = charts.exceedance_curve(
            curves,
            unit=unit,
            dark=state.dark_mode(),
            title="Exceedance curves at the horizon year",
        )
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Measure": label,
                        "Return period (years)": return_per,
                        f"Impact ({unit})": impacts,
                    }
                )
                for label, (return_per, impacts) in curves.items()
            ],
            ignore_index=True,
        )
        components.chart_with_table(
            figure,
            frame,
            table_label="Show the curve values",
            download_name="measure_exceedance.csv",
            key="cba-efc",
        )


def _combine(cost_ben) -> None:
    """Combine measures into a package, and layer risk transfer on top."""
    st.subheader("Combine measures")
    st.caption(
        "Measures are appraised one at a time, so their benefits overlap. "
        "Combining them adds the averted damage event by event, which is the "
        "honest joint figure."
    )

    names = [name for name in cost_ben.imp_meas_future if name != analysis.NO_MEASURE]
    if len(names) < 2:
        st.info("Two or more measures are needed to form a package.")
        return

    chosen = st.multiselect(
        "Measures in the package", names, default=names, key="w_cba_combine_pick"
    )
    package_name = st.text_input(
        "Package name", value="Combined package", key="w_cba_combine_name"
    )

    with st.expander("Add a risk transfer layer on top of the package"):
        attach = st.number_input(
            f"Attachment ({cost_ben.unit})",
            min_value=0.0,
            value=0.0,
            step=1.0e7,
            format="%.0f",
            help="Per-event loss above which the layer pays.",
            key="w_cba_rt_attach",
        )
        cover = st.number_input(
            f"Cover ({cost_ben.unit})",
            min_value=0.0,
            value=0.0,
            step=1.0e7,
            format="%.0f",
            help="Maximum the layer pays per event. 0 means no layer.",
            key="w_cba_rt_cover",
        )
        cost_fix = st.number_input(
            f"Fixed cost ({cost_ben.unit})",
            min_value=0.0,
            value=0.0,
            step=1.0e6,
            format="%.0f",
            key="w_cba_rt_fix",
        )
        cost_factor = st.number_input(
            "Premium factor",
            min_value=0.0,
            value=2.0,
            step=0.1,
            help="Premium as a multiple of the expected annual payout.",
            key="w_cba_rt_factor",
        )

    if not st.button("Combine", key="w_cba_combine_run"):
        _show_combined()
        return

    if len(chosen) < 2:
        st.warning("Pick at least two measures.")
        return

    disc_rates = analysis.build_disc_rates(
        float(state.get("disc_rate")),
        int(state.get("present_year")),
        int(state.get("future_year")),
    )
    risk_func = analysis.RISK_FUNCTIONS[state.get("risk_metric")]

    try:
        with st.spinner("Combining measures..."):
            combined = cost_ben.combine_measures(
                chosen,
                package_name,
                new_color=np.array([0.2, 0.5, 0.8]),
                disc_rates=disc_rates,
                imp_time_depen=float(state.get("imp_time_depen")),
                risk_func=risk_func,
            )
            if cover > 0:
                combined.apply_risk_transfer(
                    package_name,
                    float(attach),
                    float(cover),
                    disc_rates,
                    cost_fix=float(cost_fix),
                    cost_factor=float(cost_factor),
                    imp_time_depen=float(state.get("imp_time_depen")),
                    risk_func=risk_func,
                )
    except Exception as err:
        components.error_box(err, "combine the measures")
        return

    state.put("combined", combined)
    st.rerun()


def _show_combined() -> None:
    """Render the combined package, if one has been computed."""
    combined = state.get("combined")
    if combined is None:
        return

    table = analysis.cost_benefit_table(combined)
    if table.empty:
        st.info("The combination produced no ranked result.")
        return

    unit = combined.unit
    best = table.iloc[0]
    components.stat_row(
        [
            ("Package", str(best["Measure"]), None),
            ("Cost", fmt_compact(best[f"Cost ({unit})"], unit), None),
            ("Averted damage", fmt_compact(best[f"Benefit ({unit})"], unit), None),
            ("Benefit/cost", fmt_ratio(best["Benefit/cost ratio"]), None),
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    components.download_frame(table, "combined_package.csv", key="cba-combined-dl")
