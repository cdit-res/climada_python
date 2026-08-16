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

Turning a finished interface session into something portable: a summary, a
bundle of tables, and a script that reproduces the run outside the interface.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from climada.ui import analysis

__all__ = ["assumptions_table", "reproduction_script", "result_frames"]


def assumptions_table(meta: Dict[str, Any], labels: Dict[str, str]) -> pd.DataFrame:
    """The settings a reader needs in order to judge the numbers.

    Parameters
    ----------
    meta : dict
        Cost-benefit settings as stored by the interface.
    labels : dict
        Provenance strings for hazard, exposures, impact functions and measures.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``Setting`` and ``Value``.
    """
    rows: List[Dict[str, Any]] = [
        {"Setting": "Hazard", "Value": labels.get("hazard", "")},
        {"Setting": "Exposures", "Value": labels.get("exposures", "")},
        {"Setting": "Impact functions", "Value": labels.get("impf", "")},
        {"Setting": "Measures", "Value": labels.get("measures", "")},
    ]
    if meta:
        rows.extend(
            [
                {"Setting": "Present year", "Value": meta.get("present_year")},
                {"Setting": "Horizon year", "Value": meta.get("future_year")},
                {
                    "Setting": "Discount rate",
                    "Value": f"{meta.get('disc_rate', 0):.3%}",
                },
                {"Setting": "Risk metric", "Value": meta.get("risk_metric")},
                {
                    "Setting": "Impact growth path",
                    "Value": meta.get("imp_time_depen"),
                },
                {
                    "Setting": "Scenario modelled",
                    "Value": "yes" if meta.get("scenario") else "no",
                },
            ]
        )
        if meta.get("scenario"):
            rows.extend(
                [
                    {
                        "Setting": "Exposure growth factor",
                        "Value": meta.get("growth_factor"),
                    },
                    {
                        "Setting": "Hazard intensity factor",
                        "Value": meta.get("intensity_factor"),
                    },
                    {
                        "Setting": "Hazard frequency factor",
                        "Value": meta.get("frequency_factor"),
                    },
                ]
            )
    return pd.DataFrame(rows)


def result_frames(
    risk: Optional[analysis.RiskResult],
    cost_ben,
    combined,
    meta: Dict[str, Any],
    labels: Dict[str, str],
    measure_rows: List[Dict[str, Any]],
    impact_unit: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Every table the session produced, keyed by sheet name.

    Parameters
    ----------
    risk : RiskResult or None
        Physical risk result, if one was computed.
    cost_ben : climada.engine.cost_benefit.CostBenefit or None
    combined : climada.engine.cost_benefit.CostBenefit or None
        Result of combining measures, if the user did that.
    meta : dict
        Cost-benefit settings.
    labels : dict
        Provenance strings.
    measure_rows : list of dict
        The measure specifications as edited.
    impact_unit : str, optional
        Unit the impact is in, when it differs from the exposures' -- a heat
        mortality run has exposures in people but an impact in deaths.
        Default: the exposures' unit.

    Returns
    -------
    dict
        Mapping of sheet name to table. Always contains ``Assumptions``.
    """
    frames: Dict[str, pd.DataFrame] = {"Assumptions": assumptions_table(meta, labels)}

    if measure_rows:
        frames["Measures"] = pd.DataFrame(
            measure_rows, columns=list(analysis.MEASURE_COLUMNS)
        )

    if risk is not None:
        unit = impact_unit or risk.unit
        frames["Return periods"] = analysis.exceedance_table(risk)
        frames["Exceedance curve"] = pd.DataFrame(
            {
                "Return period (years)": risk.freq_curve_return_per,
                f"Impact ({unit})": risk.freq_curve_impact,
            }
        )
        frames["Top events"] = analysis.top_events_table(risk.impact, limit=50)
        frames["Impact by location"] = analysis.impact_point_table(
            risk.impact, limit=5000
        )
        frames["Risk summary"] = pd.DataFrame(
            [
                {
                    "Metric": "Total exposed value",
                    "Value": risk.total_value,
                    "Unit": risk.unit,
                },
                {
                    "Metric": "Average annual impact",
                    "Value": risk.aai,
                    "Unit": unit,
                },
                {
                    "Metric": "Annual impact / exposed value",
                    "Value": risk.loss_ratio,
                    "Unit": f"{unit} per {risk.unit}",
                },
                {
                    "Metric": "100-year impact",
                    "Value": risk.rp_value(100),
                    "Unit": unit,
                },
                {
                    "Metric": "250-year impact",
                    "Value": risk.rp_value(250),
                    "Unit": unit,
                },
                {
                    "Metric": "Largest single event",
                    "Value": risk.max_event_impact,
                    "Unit": unit,
                },
            ]
        )

    if cost_ben is not None:
        frames["Cost benefit"] = analysis.cost_benefit_table(cost_ben)
        frames["Residual risk"] = analysis.measure_risk_table(cost_ben)
        summary = analysis.cost_benefit_summary(cost_ben)
        frames["CBA summary"] = pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in summary.items()]
        )

    if combined is not None:
        frames["Combined package"] = analysis.cost_benefit_table(combined)

    return frames


def reproduction_script(
    meta: Dict[str, Any],
    labels: Dict[str, str],
    measure_rows: List[Dict[str, Any]],
    haz_type: str,
    demo_key: Optional[str] = None,
) -> str:
    """A runnable script that repeats the session's calculation.

    The script loads data the same way the session did where that is
    expressible in code -- a bundled demo, otherwise a placeholder path the
    user fills in -- then rebuilds the measures, discount rates and scenario
    exactly as configured.

    Parameters
    ----------
    meta : dict
        Cost-benefit settings.
    labels : dict
        Provenance strings, used in the header comment.
    measure_rows : list of dict
        Measure specifications.
    haz_type : str
        Hazard type of the analysis.
    demo_key : str, optional
        Key into :py:data:`climada.ui.datasets.DEMO_SCENARIOS` when the session
        used a bundled demo.

    Returns
    -------
    str
        Python source.
    """
    present = meta.get("present_year", 2025)
    future = meta.get("future_year", 2050)
    rate = meta.get("disc_rate", 0.02)
    metric = meta.get("risk_metric", "Average annual impact")
    time_dep = meta.get("imp_time_depen", 1.0)
    scenario = bool(meta.get("scenario"))
    growth = meta.get("growth_factor", 1.0)
    intensity = meta.get("intensity_factor", 1.0)
    frequency = meta.get("frequency_factor", 1.0)

    risk_func = {
        "Average annual impact": "risk_aai_agg",
        "100-year return period impact": "risk_rp_100",
        "250-year return period impact": "risk_rp_250",
    }.get(metric, "risk_aai_agg")

    if demo_key:
        loading = (
            "from climada.ui.datasets import load_demo\n\n"
            f"bundle = load_demo({demo_key!r})\n"
            "hazard = bundle['hazard']\n"
            "exposures = bundle['entity'].exposures\n"
            "impf_set = bundle['entity'].impact_funcs\n"
        )
    else:
        loading = (
            "from climada.entity import Exposures, ImpactFuncSet\n"
            "from climada.hazard import Hazard\n\n"
            "# Point these at the files the interface session used:\n"
            f"#   hazard:    {labels.get('hazard', '')}\n"
            f"#   exposures: {labels.get('exposures', '')}\n"
            f"#   curves:    {labels.get('impf', '')}\n"
            "hazard = Hazard.from_hdf5('hazard.h5')\n"
            "exposures = Exposures.from_hdf5('exposures.h5')\n"
            "impf_set = ImpactFuncSet.from_excel('impact_functions.xlsx')\n"
        )

    measures_source = "MEASURES = [\n"
    for row in measure_rows:
        measures_source += (
            "    " + repr({key: row[key] for key in analysis.MEASURE_COLUMNS}) + ",\n"
        )
    measures_source += "]\n"

    scenario_source = ""
    if scenario:
        scenario_source = (
            "\n# Scenario at the horizon year\n"
            f"future_exposures = grow_exposures(exposures, {growth!r}, ref_year=FUTURE_YEAR)\n"
            "ent_future = build_entity(\n"
            "    future_exposures, impf_set, measure_set, disc_rates, ref_year=FUTURE_YEAR\n"
            ")\n"
        )
        if intensity != 1.0 or frequency != 1.0:
            scenario_source += (
                f"haz_future = scale_hazard(hazard, {intensity!r}, {frequency!r})\n"
            )
        else:
            scenario_source += "haz_future = None\n"
    else:
        scenario_source = "\nent_future = None\nhaz_future = None\n"

    return f'''"""Reproduces a CLIMADA cost-benefit analysis run in the CLIMADA UI.

Hazard:          {labels.get('hazard', 'n/a')}
Exposures:       {labels.get('exposures', 'n/a')}
Impact function: {labels.get('impf', 'n/a')}
Measures:        {labels.get('measures', 'n/a')}
"""

from climada.engine.cost_benefit import risk_aai_agg, risk_rp_100, risk_rp_250
from climada.ui.analysis import (
    build_disc_rates,
    build_entity,
    build_measure_set,
    compute_risk,
    cost_benefit_table,
    grow_exposures,
    run_cost_benefit,
    scale_hazard,
)

{loading}
PRESENT_YEAR = {present}
FUTURE_YEAR = {future}
HAZ_TYPE = {haz_type!r}

{measures_source}
# Physical risk today
risk = compute_risk(exposures, impf_set, hazard)
print(f"Average annual impact: {{risk.aai:,.0f}} {{risk.unit}}")
print(f"100-year loss:         {{risk.rp_value(100):,.0f}} {{risk.unit}}")

# Cost-benefit analysis
measure_set = build_measure_set(MEASURES, HAZ_TYPE)
disc_rates = build_disc_rates({rate!r}, PRESENT_YEAR, FUTURE_YEAR)
entity = build_entity(
    exposures, impf_set, measure_set, disc_rates, ref_year=PRESENT_YEAR
)
{scenario_source}
cost_ben = run_cost_benefit(
    hazard,
    entity,
    haz_future=haz_future,
    ent_future=ent_future,
    future_year=FUTURE_YEAR,
    risk_func={risk_func},
    imp_time_depen={time_dep!r},
)
print(cost_benefit_table(cost_ben).to_string(index=False))
'''
