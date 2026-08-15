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

Report page: take the session's results out of the browser.
"""

# The view layer is the boundary between CLIMADA and the browser: any
# failure below must become a message on the page, never a crashed session.
# pylint: disable=broad-exception-caught


from typing import Dict

import streamlit as st

from climada.ui import analysis, components, datasets, report, state
from climada.ui.formatting import fmt_compact, fmt_percent, fmt_ratio


def render() -> None:
    """Draw the report page."""
    components.page_header(
        "Report",
        "The assumptions behind the numbers, every table the session produced, "
        "and a script that reproduces the run without the interface.",
    )

    risk = state.get("risk")
    cost_ben = state.get("cost_benefit")
    if risk is None and cost_ben is None:
        st.info(
            "Nothing to report yet. Run the **Risk** page, and the "
            "**Cost-benefit** page if you have measures."
        )
        return

    _narrative(risk, cost_ben)
    st.divider()
    _assumptions()
    st.divider()
    _downloads(risk, cost_ben)
    st.divider()
    _script()


def _narrative(risk, cost_ben) -> None:
    """A plain-language summary of what was found."""
    st.subheader("Summary")
    lines = []

    if risk is not None:
        lines.append(
            f"Against the loaded {risk.haz_type} event set, the exposures carry "
            f"an average annual impact of **{fmt_compact(risk.aai, risk.unit)}**, "
            f"or {fmt_percent(risk.loss_ratio, digits=3)} of the "
            f"{fmt_compact(risk.total_value, risk.unit)} exposed. A one-in-100-year "
            f"year costs {fmt_compact(risk.rp_value(100), risk.unit)}; the worst "
            f"single event in the set costs "
            f"{fmt_compact(risk.max_event_impact, risk.unit)}."
        )

    if cost_ben is not None:
        summary = analysis.cost_benefit_summary(cost_ben)
        table = analysis.cost_benefit_table(cost_ben)
        years = summary["future_year"] - summary["present_year"] + 1
        lines.append(
            f"Over {years} years to {summary['future_year']}, the discounted "
            f"climate risk is "
            f"**{fmt_compact(summary['total_climate_risk'], cost_ben.unit)}**. The "
            f"measures as specified avert "
            f"{fmt_compact(summary['averted'], cost_ben.unit)} for "
            f"{fmt_compact(summary['total_cost'], cost_ben.unit)}, a portfolio "
            f"benefit/cost ratio of {fmt_ratio(summary['portfolio_bcr'])}."
        )
        paying = table[table["Benefit/cost ratio"] >= 1]
        if not paying.empty:
            names = ", ".join(str(name) for name in paying["Measure"])
            lines.append(
                f"Measures that pay for themselves at these assumptions: **{names}**."
            )
        else:
            lines.append(
                "**No measure pays for itself** at these assumptions. Either the "
                "costs are too high for the risk being averted, or the horizon "
                "is too short for the benefits to accumulate."
            )

    for line in lines:
        st.markdown(line)


def _assumptions() -> None:
    """The settings table."""
    st.subheader("Assumptions")
    st.dataframe(
        report.assumptions_table(state.get("cost_benefit_meta") or {}, _labels()),
        use_container_width=True,
        hide_index=True,
    )


def _labels() -> Dict[str, str]:
    """Provenance strings for the current inputs."""
    rows = state.measure_rows()
    return {
        "hazard": state.get("hazard_label") or "not loaded",
        "exposures": state.get("exposures_label") or "not loaded",
        "impf": state.get("impf_label") or "not loaded",
        "measures": (
            f"{len(rows)} measure(s) from {state.get('measure_source') or 'the interface'}"
            if rows
            else "none"
        ),
    }


def _downloads(risk, cost_ben) -> None:
    """CSV and Excel exports of every result table."""
    st.subheader("Download the results")
    frames = report.result_frames(
        risk,
        cost_ben,
        state.get("combined"),
        state.get("cost_benefit_meta") or {},
        _labels(),
        state.measure_rows(),
    )

    try:
        workbook = components.excel_bytes(frames)
    except Exception as err:
        components.error_box(err, "build the Excel workbook")
    else:
        st.download_button(
            "Download everything as one Excel workbook",
            data=workbook,
            file_name="climada_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            key="w_export_xlsx",
        )

    st.markdown("**Individual tables**")
    for name, frame in frames.items():
        with st.expander(f"{name} ({len(frame):,} rows)"):
            st.dataframe(frame.head(200), use_container_width=True, hide_index=True)
            components.download_frame(
                frame,
                f"{name.lower().replace(' ', '_')}.csv",
                key=f"w_export_{name}",
            )


def _script() -> None:
    """The reproduction script."""
    st.subheader("Reproduce this run in code")
    st.caption(
        "The interface is a shell over `climada.ui.analysis`. This script calls "
        "the same functions with the same settings, so a result you found here "
        "can go into a pipeline unchanged."
    )

    demo_key = None
    label = state.get("hazard_label")
    for key, scenario in datasets.DEMO_SCENARIOS.items():
        if scenario.label == label:
            demo_key = key
            break

    source = report.reproduction_script(
        state.get("cost_benefit_meta") or {},
        _labels(),
        state.measure_rows(),
        state.haz_type(),
        demo_key=demo_key,
    )
    st.code(source, language="python")
    st.download_button(
        "Download the script",
        data=source.encode("utf-8"),
        file_name="climada_analysis.py",
        mime="text/x-python",
        key="w_export_script",
    )
