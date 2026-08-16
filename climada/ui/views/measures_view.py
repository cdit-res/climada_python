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

Measures page: define the adaptation options the cost-benefit analysis ranks.
"""

# The view layer is the boundary between CLIMADA and the browser: any
# failure below must become a message on the page, never a crashed session.
# pylint: disable=broad-exception-caught


from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from climada.ui import analysis, charts, components, datasets, heat, state
from climada.ui.formatting import fmt_compact, fmt_percent

PRESETS: Dict[str, Dict[str, Any]] = {
    "Coastal protection": {
        "note": "Cuts the hazard intensity that reaches the assets by a fixed "
        "amount -- a negative offset shifts the vulnerability curve right.",
        "fields": {"hazard_inten_imp_b": -4.0},
    },
    "Building retrofit": {
        "note": "Reduces the damage a given intensity causes, by a fixed share.",
        "fields": {"mdd_impact_a": 0.75},
    },
    "Reduce exposure share": {
        "note": "Reduces the fraction of assets affected at a given intensity.",
        "fields": {"paa_impact_a": 0.7},
    },
    "Insurance layer": {
        "note": "Transfers a slice of the loss, between attachment and cover, to a "
        "counterparty for a premium.",
        "fields": {
            "risk_transf_attach": 5.0e8,
            "risk_transf_cover": 2.0e9,
            "risk_transf_cost_factor": 2.0,
        },
    },
}

COLUMN_HELP = {
    "name": "Unique name for the measure.",
    "cost": "Total discounted cost of implementing the measure, in the exposure "
    "value unit. CLIMADA does not discount this for you.",
    "hazard_inten_imp_a": "Multiplier on the intensity axis of the vulnerability "
    "curve. Leave at 1 unless scaling proportionally.",
    "hazard_inten_imp_b": "Offset subtracted from the vulnerability curve's "
    "intensity axis. NEGATIVE values are protection: -4 means the assets behave "
    "as though the hazard were 4 units weaker.",
    "mdd_impact_a": "Multiplier on the mean damage degree. 0.75 means the same "
    "event does 25% less damage.",
    "mdd_impact_b": "Constant added to the mean damage degree.",
    "paa_impact_a": "Multiplier on the percentage of assets affected.",
    "paa_impact_b": "Constant added to the percentage of assets affected.",
    "hazard_freq_cutoff": "Frequency above which events are treated as fully "
    "prevented. 0 disables the cutoff.",
    "risk_transf_attach": "Loss per event above which the risk transfer pays out.",
    "risk_transf_cover": "Maximum the risk transfer pays per event. 0 disables it.",
    "risk_transf_cost_factor": "Premium as a multiple of the expected payout.",
}


def render() -> None:
    """Draw the measures page."""
    components.page_header(
        "Adaptation measures",
        "Each measure changes the hazard reaching the assets, the damage a given "
        "intensity causes, or who carries the loss. Costs are entered as "
        "already-discounted totals.",
    )

    _editor()
    st.divider()
    _presets()
    st.divider()
    _preview()


def _editor() -> None:
    """The editable measure table."""
    st.subheader("Measures")
    rows = state.measure_rows()
    if not rows:
        st.info(
            "No measures yet. Add one below, load a preset, or import a demo "
            "scenario on the **Data** page."
        )

    frame = pd.DataFrame(rows, columns=list(analysis.MEASURE_COLUMNS))
    unit = state.value_unit()
    haz_units = state.get("hazard").units if state.get("hazard") is not None else ""

    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        use_container_width=True,
        key="w_meas_editor",
        column_config={
            "name": st.column_config.TextColumn("Name", help=COLUMN_HELP["name"]),
            "cost": st.column_config.NumberColumn(
                f"Cost ({unit})", format="%.0f", help=COLUMN_HELP["cost"]
            ),
            "hazard_inten_imp_a": st.column_config.NumberColumn(
                "Intensity x", format="%.3f", help=COLUMN_HELP["hazard_inten_imp_a"]
            ),
            "hazard_inten_imp_b": st.column_config.NumberColumn(
                f"Intensity offset{' (' + haz_units + ')' if haz_units else ''}",
                format="%.3f",
                help=COLUMN_HELP["hazard_inten_imp_b"],
            ),
            "mdd_impact_a": st.column_config.NumberColumn(
                "Damage x", format="%.3f", help=COLUMN_HELP["mdd_impact_a"]
            ),
            "mdd_impact_b": st.column_config.NumberColumn(
                "Damage +", format="%.3f", help=COLUMN_HELP["mdd_impact_b"]
            ),
            "paa_impact_a": st.column_config.NumberColumn(
                "Affected x", format="%.3f", help=COLUMN_HELP["paa_impact_a"]
            ),
            "paa_impact_b": st.column_config.NumberColumn(
                "Affected +", format="%.3f", help=COLUMN_HELP["paa_impact_b"]
            ),
            "hazard_freq_cutoff": st.column_config.NumberColumn(
                "Frequency cutoff",
                format="%.4f",
                help=COLUMN_HELP["hazard_freq_cutoff"],
            ),
            "risk_transf_attach": st.column_config.NumberColumn(
                f"Attachment ({unit})",
                format="%.0f",
                help=COLUMN_HELP["risk_transf_attach"],
            ),
            "risk_transf_cover": st.column_config.NumberColumn(
                f"Cover ({unit})", format="%.0f", help=COLUMN_HELP["risk_transf_cover"]
            ),
            "risk_transf_cost_factor": st.column_config.NumberColumn(
                "Premium factor",
                format="%.2f",
                help=COLUMN_HELP["risk_transf_cost_factor"],
            ),
        },
    )

    left, right = st.columns([1, 4])
    if left.button("Save measures", type="primary", key="w_meas_save"):
        cleaned = _clean(edited)
        problems = _validate(cleaned)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            state.set_measure_rows(cleaned)
            state.put("measure_source", "edited in the interface")
            right.success(f"Saved {len(cleaned)} measure(s).")
            st.rerun()

    with st.expander("How the parameters work"):
        st.markdown(
            "CLIMADA applies a measure by rewriting the vulnerability curve:\n\n"
            "- the intensity axis becomes "
            "`intensity x (Intensity x) - (Intensity offset)`. **A negative "
            "offset is protection**, because it shifts the curve to the right: "
            "a sea wall that keeps 4 m/s of wind off the assets is "
            "`Intensity offset = -4`, and a positive offset makes things worse;\n"
            "- the mean damage degree becomes `mdd x (Damage x) + (Damage +)`, so a "
            "retrofit that cuts damage by a quarter is `Damage x = 0.75`;\n"
            "- the affected share becomes `paa x (Affected x) + (Affected +)`.\n\n"
            "Risk transfer is layered on top of whatever the measure leaves: the "
            "counterparty pays the part of each event's loss between the "
            "attachment and the cover, and charges the premium factor times the "
            "expected payout."
        )


def _clean(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Drop blank rows and coerce the editor's output back to plain records."""
    frame = frame.copy()
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame = frame[frame["name"] != ""]

    defaults = analysis.default_measure_row()
    records: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = dict(defaults)
        record["name"] = row["name"]
        for column in analysis.MEASURE_COLUMNS[1:]:
            value = row.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            record[column] = float(value)
        records.append(record)
    return records


def _validate(rows: List[Dict[str, Any]]) -> List[str]:
    """Catch the mistakes that would otherwise surface as opaque errors."""
    problems: List[str] = []
    names = [row["name"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(
            "Measure names must be unique; repeated: " + ", ".join(duplicates)
        )

    for row in rows:
        if row["cost"] < 0:
            problems.append(f"'{row['name']}' has a negative cost.")
        if (
            row["risk_transf_cover"] > 0
            and row["risk_transf_cover"] <= row["risk_transf_attach"]
        ):
            problems.append(
                f"'{row['name']}' has a risk transfer cover at or below its "
                "attachment, so the layer is empty."
            )
        if row["hazard_inten_imp_a"] == 0:
            problems.append(
                f"'{row['name']}' multiplies the intensity axis by zero, which "
                "collapses the vulnerability curve."
            )
    return problems


def _presets() -> None:
    """One-click starting points, matched to the loaded hazard."""
    st.subheader("Add a preset")
    presets = (
        heat.HEAT_MEASURE_PRESETS if state.haz_type() == heat.HAZ_TYPE else PRESETS
    )
    if presets is heat.HEAT_MEASURE_PRESETS:
        st.caption(
            "Heat adaptation options. The reductions are indicative and vary "
            "widely by city, housing stock and population age -- replace them "
            "with local evidence."
        )

    columns = st.columns(len(presets))
    for column, (label, preset) in zip(columns, presets.items()):
        with column:
            st.markdown(f"**{label}**")
            st.caption(preset["note"])
            if st.button("Add", key=f"w_meas_preset_{label}"):
                rows = state.measure_rows()
                name = _unique_name(label, [row["name"] for row in rows])
                row = analysis.default_measure_row(name)
                row.update(preset["fields"])
                rows.append(row)
                state.set_measure_rows(rows)
                st.rerun()


def _unique_name(base: str, existing: List[str]) -> str:
    """Append a counter until the name is free."""
    if base not in existing:
        return base
    index = 2
    while f"{base} {index}" in existing:
        index += 1
    return f"{base} {index}"


def _preview() -> None:
    """Show what one measure does to the curve and to the annual risk."""
    st.subheader("Preview a measure")
    rows = state.measure_rows()
    if not rows:
        return
    if not state.inputs_ready():
        st.info("Load hazard, exposures and vulnerability curves to preview effects.")
        return

    names = [row["name"] for row in rows]
    chosen = st.selectbox("Measure", names, key="w_meas_preview_pick")
    row = next(row for row in rows if row["name"] == chosen)

    hazard = state.get("hazard")
    exposures = state.get("exposures")
    impf_set = state.get("impf_set")
    haz_type = hazard.haz_type

    measure = analysis.build_measure(row, haz_type)
    try:
        new_exp, new_impf, new_haz = analysis.measure_impact_curves(
            measure, exposures, impf_set, hazard
        )
    except Exception as err:
        components.error_box(err, "apply the measure")
        return

    before = datasets.impf_curves(impf_set, haz_type)
    after = datasets.impf_curves(new_impf, haz_type)
    if before and after:
        first_before = next(iter(before.items()))
        first_after = next(iter(after.items()))
        figure = charts.impact_function_chart(
            {
                "Without the measure": first_before[1],
                f"With '{chosen}'": first_after[1],
            },
            intensity_unit=datasets.impf_intensity_unit(impf_set, haz_type),
            dark=state.dark_mode(),
            title=f"Vulnerability curve {first_before[0]}",
        )
        frame = pd.DataFrame(
            {
                "Intensity (without)": first_before[1]["intensity"],
                "Damage ratio (without)": first_before[1]["mdr"],
                "Intensity (with)": first_after[1]["intensity"],
                "Damage ratio (with)": first_after[1]["mdr"],
            }
        )
        components.chart_with_table(
            figure,
            frame,
            table_label="Show the curve values",
            download_name="measure_curve.csv",
            key="meas-curve",
        )

    if st.button(
        "Compute the annual risk under this measure", key="w_meas_preview_run"
    ):
        try:
            with st.spinner("Computing..."):
                baseline = analysis.compute_risk(exposures, impf_set, hazard)
                treated = analysis.compute_risk(new_exp, new_impf, new_haz)
        except Exception as err:
            components.error_box(err, "compute the impact under the measure")
            return

        averted = baseline.aai - treated.aai
        share = averted / baseline.aai if baseline.aai else float("nan")

        # A heat run has exposures in people but an impact in deaths, so the
        # exposure unit is the wrong label for these numbers.
        metric_key = state.get("heat_metric")
        unit = baseline.unit
        if metric_key and haz_type == heat.HAZ_TYPE:
            unit = heat.METRICS[metric_key].unit

        components.stat_row(
            [
                ("Annual risk without", fmt_compact(baseline.aai, unit), None),
                ("Annual risk with", fmt_compact(treated.aai, unit), None),
                ("Annual risk averted", fmt_compact(averted, unit), None),
                ("Share averted", fmt_percent(share), None),
            ]
        )
        st.caption(
            "This is the annual effect only. The cost-benefit page discounts it "
            "over the appraisal horizon and compares it with the cost."
        )
