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

Session state for the CLIMADA user interface.

The interface keeps one analysis in flight at a time. Inputs live here;
results are cleared whenever an input they depend on changes, so the screen
never shows numbers computed from data the user has since replaced.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from climada.ui import analysis

__all__ = ["dark_mode", "ensure_defaults", "get", "invalidate", "put", "state_summary"]


DEFAULTS: Dict[str, Any] = {
    # Inputs
    "hazard": None,
    "hazard_label": "",
    "exposures": None,
    "exposures_label": "",
    "impf_set": None,
    "impf_label": "",
    "impf_note": "",
    "heat_metric": None,
    "measure_rows": [],
    "measure_source": "",
    # Appraisal settings
    "present_year": 2025,
    "future_year": 2050,
    "disc_rate": 0.02,
    "risk_metric": "Average annual impact",
    "imp_time_depen": 1.0,
    "intensity_factor": 1.0,
    "frequency_factor": 1.0,
    "growth_factor": 1.0,
    "use_future": True,
    "future_hazard": None,
    "future_hazard_label": "",
    # Results
    "risk": None,
    "cost_benefit": None,
    "cost_benefit_meta": None,
    "waterfall": None,
    "combined": None,
    # Presentation
    "basemap": False,
    "map_point_limit": 5000,
}
"""Initial value of every session key the interface uses."""

RESULT_KEYS = ("risk", "cost_benefit", "cost_benefit_meta", "waterfall", "combined")
"""Keys holding computed output, dropped whenever an input changes."""


def ensure_defaults() -> None:
    """Populate any session key that has not been set yet.

    Safe to call on every rerun; existing values are left alone.
    """
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value


def get(key: str, default: Any = None) -> Any:
    """Read a session value.

    Parameters
    ----------
    key : str
    default : Any, optional
        Returned when the key is unset.

    Returns
    -------
    Any
    """
    return st.session_state.get(key, DEFAULTS.get(key, default))


def put(key: str, value: Any, invalidates: bool = False) -> None:
    """Write a session value, optionally dropping dependent results.

    Parameters
    ----------
    key : str
    value : Any
    invalidates : bool, optional
        Clear computed results as well, because they were derived from the old
        value. Default: False.
    """
    st.session_state[key] = value
    if invalidates:
        invalidate()


def invalidate() -> None:
    """Drop every computed result, leaving the inputs in place."""
    for key in RESULT_KEYS:
        st.session_state[key] = DEFAULTS[key]


def dark_mode() -> bool:
    """Whether the viewer is currently in dark mode.

    Falls back to light when Streamlit cannot report a theme, which is the
    safer default: the light palette carries visible labels and table views.

    Returns
    -------
    bool
    """
    try:
        return str(st.context.theme.type).lower() == "dark"
    # Theme reporting varies by Streamlit version and transport; a missing
    # theme must never stop a page from rendering.
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def haz_type() -> str:
    """Hazard type of the loaded hazard, or an empty string.

    Returns
    -------
    str
    """
    hazard = get("hazard")
    return hazard.haz_type if hazard is not None else ""


def value_unit() -> str:
    """Value unit of the loaded exposures, defaulting to USD.

    Returns
    -------
    str
    """
    exposures = get("exposures")
    if exposures is None:
        return "USD"
    return exposures.value_unit or "USD"


def inputs_ready() -> bool:
    """Whether hazard, exposures and impact functions are all loaded.

    Returns
    -------
    bool
    """
    return all(get(key) is not None for key in ("hazard", "exposures", "impf_set"))


def measure_rows() -> List[Dict[str, Any]]:
    """The measure specifications currently defined.

    Returns
    -------
    list of dict
    """
    return list(get("measure_rows") or [])


def set_measure_rows(rows: List[Dict[str, Any]]) -> None:
    """Replace the measure specifications and drop dependent results.

    Parameters
    ----------
    rows : list of dict
    """
    st.session_state["measure_rows"] = list(rows)
    st.session_state["cost_benefit"] = None
    st.session_state["cost_benefit_meta"] = None
    st.session_state["combined"] = None


def adopt_entity(entity, label: str, keep_measures: bool = True) -> None:
    """Load exposures, impact functions, measures and rates from an entity.

    Parameters
    ----------
    entity : climada.entity.Entity
        Entity to take the inputs from.
    label : str
        Human-readable provenance, shown in the interface.
    keep_measures : bool, optional
        Also import the entity's measures. Default: True.
    """
    put("exposures", entity.exposures)
    put("exposures_label", label)
    put("impf_set", entity.impact_funcs)
    put("impf_label", label)
    put("impf_note", "")
    # An entity's curves are not a heat metric; a metric left over from an
    # earlier run would relabel this one's results.
    put("heat_metric", None)
    put("present_year", int(entity.exposures.ref_year))

    rates = entity.disc_rates
    if rates is not None and len(getattr(rates, "rates", [])) > 0:
        put("disc_rate", float(rates.rates[0]))

    if keep_measures:
        rows = []
        for measure in _all_measures(entity):
            row = analysis.default_measure_row(measure.name)
            row.update(
                {
                    "cost": float(measure.cost),
                    "hazard_inten_imp_a": float(measure.hazard_inten_imp[0]),
                    "hazard_inten_imp_b": float(measure.hazard_inten_imp[1]),
                    "mdd_impact_a": float(measure.mdd_impact[0]),
                    "mdd_impact_b": float(measure.mdd_impact[1]),
                    "paa_impact_a": float(measure.paa_impact[0]),
                    "paa_impact_b": float(measure.paa_impact[1]),
                    "hazard_freq_cutoff": float(measure.hazard_freq_cutoff),
                    "risk_transf_attach": float(measure.risk_transf_attach),
                    "risk_transf_cover": float(measure.risk_transf_cover),
                    "risk_transf_cost_factor": float(measure.risk_transf_cost_factor),
                }
            )
            rows.append(row)
        set_measure_rows(rows)
        put("measure_source", label)

    invalidate()


def _all_measures(entity) -> List[Any]:
    """Every measure in an entity, across hazard types."""
    measures = entity.measures.get_measure()
    if isinstance(measures, dict):
        return [
            measure for by_name in measures.values() for measure in by_name.values()
        ]
    return list(measures)


def state_summary() -> Dict[str, Optional[str]]:
    """One-line descriptions of what is currently loaded.

    Returns
    -------
    dict
        Keys ``hazard``, ``exposures``, ``impact functions`` and ``measures``,
        with ``None`` where nothing is loaded.
    """
    hazard = get("hazard")
    exposures = get("exposures")
    impf_set = get("impf_set")
    rows = measure_rows()

    summary: Dict[str, Optional[str]] = {}
    summary["hazard"] = (
        f"{hazard.haz_type} - {hazard.size:,} events - {get('hazard_label')}"
        if hazard is not None
        else None
    )
    summary["exposures"] = (
        f"{len(exposures.gdf):,} points - {get('exposures_label')}"
        if exposures is not None
        else None
    )
    summary["impact functions"] = (
        f"{impf_set.size():,} curves - {get('impf_label')}"
        if impf_set is not None
        else None
    )
    summary["measures"] = f"{len(rows)} defined" if rows else None
    return summary
