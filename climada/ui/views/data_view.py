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

Data page: choose the hazard, the exposures and the vulnerability curves.
"""

# The view layer is the boundary between CLIMADA and the browser: any
# failure below must become a message on the page, never a crashed session.
# pylint: disable=broad-exception-caught


import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from climada.ui import charts, components, datasets, heat, state
from climada.ui.formatting import fmt_compact

UPLOAD_DIR = Path(tempfile.gettempdir()) / "climada_ui_uploads"


def render() -> None:
    """Draw the data page."""
    components.page_header(
        "Data",
        "Assemble the three ingredients of a risk calculation: what can happen "
        "(hazard), what is at stake (exposures) and how badly it is damaged "
        "(vulnerability).",
    )

    _quick_start()
    st.divider()

    hazard_tab, exposures_tab, impf_tab = st.tabs(
        ["Hazard", "Exposures", "Vulnerability"]
    )
    with hazard_tab:
        _hazard_section()
    with exposures_tab:
        _exposures_section()
    with impf_tab:
        _impf_section()

    st.divider()
    _readiness_section()


# --------------------------------------------------------------------------- #
# Quick start
# --------------------------------------------------------------------------- #


def _quick_start() -> None:
    """Load a complete bundled scenario in one click."""
    st.subheader("Start from a demo")

    generated_tab, bundled_tab = st.tabs(["Heat (generated)", "Bundled CLIMADA demos"])
    with generated_tab:
        _generated_start()
    with bundled_tab:
        _bundled_start()


def _generated_start() -> None:
    """Build a heat analysis from scratch, since none ships with CLIMADA."""
    keys = list(datasets.GENERATED_SCENARIOS)
    choice = st.selectbox(
        "Generated scenario",
        keys,
        format_func=lambda key: datasets.GENERATED_SCENARIOS[key]["label"],
        key="w_gen_choice",
    )
    st.caption(datasets.GENERATED_SCENARIOS[choice]["description"])

    metric = st.selectbox(
        "What to count",
        list(heat.METRICS),
        format_func=lambda key: heat.METRICS[key].label,
        help="Heat risk can be counted as deaths, exposure, degree-days or "
        "lost work. They use the same machinery and read very differently.",
        key="w_gen_metric",
    )
    st.caption(heat.METRICS[metric].description)

    if st.button("Generate scenario", type="primary", key="w_gen_load"):
        try:
            with st.spinner("Generating the temperature field..."):
                bundle = heat.demo_bundle(metric=metric)
        except Exception as err:
            components.error_box(err, "generate the heat scenario")
            return

        label = datasets.GENERATED_SCENARIOS[choice]["label"]
        state.put("hazard", bundle["hazard"], invalidates=True)
        state.put("hazard_label", label)
        state.put("exposures", bundle["exposures"])
        state.put("exposures_label", f"{label} (synthetic population)")
        state.put("impf_set", bundle["impf_set"])
        state.put("impf_label", heat.METRICS[metric].label)
        state.put("impf_note", bundle["note"])
        state.put("heat_metric", metric)
        state.put("present_year", int(bundle["exposures"].ref_year))
        state.put("future_year", 2050)
        state.set_measure_rows([])
        st.success(
            f"Generated '{label}'. Open **Risk** to see the "
            f"{heat.METRICS[metric].annual_noun}."
        )


def _bundled_start() -> None:
    """Load a complete scenario from the data bundled with CLIMADA."""
    available = {
        key: scenario
        for key, scenario in datasets.DEMO_SCENARIOS.items()
        if scenario.available
    }
    if not available:
        st.warning(
            "No demo data found in this CLIMADA installation. Load your own "
            "files in the tabs below."
        )
        return

    keys = list(available)
    choice = st.selectbox(
        "Demo scenario",
        keys,
        format_func=lambda key: available[key].label,
        key="w_demo_choice",
    )
    st.caption(available[choice].description)

    load_measures = st.checkbox(
        "Also load the demo's adaptation measures and discount rate",
        value=True,
        key="w_demo_measures",
    )
    if st.button("Load demo scenario", type="primary", key="w_demo_load"):
        try:
            with st.spinner("Reading demo data..."):
                bundle = datasets.load_demo(choice)
        except Exception as err:
            components.error_box(err, "load the demo scenario")
            return

        scenario = bundle["scenario"]
        state.put("hazard", bundle["hazard"])
        state.put("hazard_label", scenario.label)
        state.adopt_entity(
            bundle["entity"], scenario.label, keep_measures=load_measures
        )
        if bundle["entity_future"] is not None:
            state.put("growth_factor", 1.0)
        state.put("future_year", max(state.get("present_year") + 25, 2050))
        st.success(f"Loaded '{scenario.label}'. Open **Risk** to see the results.")


# --------------------------------------------------------------------------- #
# Hazard
# --------------------------------------------------------------------------- #


def _hazard_section() -> None:
    """Hazard loading: file upload or the CLIMADA Data API."""
    hazard = state.get("hazard")
    if hazard is not None:
        _hazard_summary(hazard)

    file_tab, grid_tab, api_tab = st.tabs(
        ["From a file", "Gridded temperature (heat)", "From the CLIMADA Data API"]
    )

    with grid_tab:
        _gridded_hazard()

    with file_tab:
        st.markdown(
            "CLIMADA hazard sets are HDF5 files (`.h5`) written by "
            "`Hazard.write_hdf5`, or spreadsheets in CLIMADA's hazard template "
            "format."
        )
        uploaded = st.file_uploader(
            "Hazard file", type=["h5", "hdf5", "xls", "xlsx"], key="w_haz_upload"
        )
        typed = st.text_input(
            "Hazard type",
            value="",
            placeholder="e.g. TC -- only needed for spreadsheets",
            key="w_haz_type",
        )
        if uploaded is not None and st.button("Load hazard", key="w_haz_load"):
            try:
                path = components.save_upload(uploaded, UPLOAD_DIR)
                with st.spinner("Reading hazard..."):
                    loaded = datasets.load_hazard_file(path, typed or None)
            except Exception as err:
                components.error_box(err, "load the hazard")
            else:
                state.put("hazard", loaded, invalidates=True)
                state.put("hazard_label", uploaded.name)
                st.success(f"Loaded {loaded.size:,} {loaded.haz_type} events.")
                st.rerun()

    with api_tab:
        _api_hazard()


def _gridded_hazard() -> None:
    """Read daily temperature fields into a heat hazard."""
    st.markdown(
        "Neither CLIMADA's Data API nor Petals serves a heat hazard, so heat "
        "risk starts from your own gridded temperature: daily maximum "
        "temperature from a reanalysis such as ERA5, or from a climate model. "
        "Every time step becomes one event."
    )
    uploaded = st.file_uploader(
        "NetCDF or GRIB file", type=["nc", "nc4", "grib"], key="w_grid_upload"
    )
    if uploaded is None:
        st.caption(
            "No file? The **Heat (generated)** demo above builds a synthetic "
            "temperature field so you can walk the workflow first."
        )
        return

    left, right = st.columns(2)
    variable = left.text_input(
        "Intensity variable",
        value="tasmax",
        help="Name of the temperature variable in the file, e.g. 'tasmax' "
        "(CMIP6) or 't2m' (ERA5).",
        key="w_grid_var",
    )
    unit = right.text_input("Intensity unit", value="degC", key="w_grid_unit")
    to_celsius = st.checkbox(
        "Convert from kelvin",
        value=False,
        help="ERA5 and most model output are in kelvin. Subtracts 273.15.",
        key="w_grid_kelvin",
    )

    with st.expander("Coordinate names"):
        cols = st.columns(3)
        time_var = cols[0].text_input("Time", value="time", key="w_grid_time")
        lat_var = cols[1].text_input("Latitude", value="latitude", key="w_grid_lat")
        lon_var = cols[2].text_input("Longitude", value="longitude", key="w_grid_lon")

    years = st.number_input(
        "Record length in years (0 = read it from the timestamps)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Sets the event frequency. A daily record spanning N years gives "
        "every day a frequency of 1/N per year, which is what makes the "
        "average annual impact come out per year.",
        key="w_grid_years",
    )

    if not st.button("Load gridded hazard", type="primary", key="w_grid_load"):
        return

    try:
        path = components.save_upload(uploaded, UPLOAD_DIR)
        with st.spinner("Reading the grid -- large files take a while..."):
            hazard = datasets.load_gridded_hazard(
                path,
                intensity=variable,
                intensity_unit=unit,
                time_var=time_var,
                lat_var=lat_var,
                lon_var=lon_var,
                to_celsius=to_celsius,
                years=float(years) or None,
            )
    except Exception as err:
        components.error_box(err, "read the gridded hazard")
        return

    state.put("hazard", hazard, invalidates=True)
    state.put("hazard_label", f"{uploaded.name} ({variable})")
    summary = heat.hazard_temperature_summary(hazard)
    st.success(
        f"Loaded {hazard.size:,} time steps over {summary['centroids']:,.0f} "
        f"grid points, spanning {summary['years']:,.1f} years."
    )
    st.rerun()


def _hazard_summary(hazard) -> None:
    """Stat tiles and an intensity note for the loaded hazard."""
    intensity = hazard.intensity
    max_intensity = float(intensity.max()) if intensity.nnz else 0.0
    years = ""
    try:
        dates = np.asarray(hazard.date)
        if dates.size and dates.max() > 0:
            years = (
                f"{pd.Timestamp.fromordinal(int(dates.min())).year}"
                f"-{pd.Timestamp.fromordinal(int(dates.max())).year}"
            )
    except (ValueError, OverflowError, TypeError):
        years = ""

    components.stat_row(
        [
            ("Hazard type", hazard.haz_type or "unknown", None),
            ("Events", f"{hazard.size:,}", "Number of events in the set"),
            ("Centroids", f"{hazard.centroids.size:,}", None),
            (
                "Peak intensity",
                f"{max_intensity:,.1f} {hazard.units}",
                "Largest intensity anywhere in the set",
            ),
        ]
    )
    caption = f"Source: {state.get('hazard_label') or 'unknown'}"
    if years:
        caption += f" - events dated {years}"
    st.caption(caption)


def _api_hazard() -> None:
    """Browse and download a hazard from the CLIMADA Data API."""
    st.markdown(
        "The Data API serves published hazard sets. Downloads are cached in "
        "your CLIMADA data directory, so the first fetch is the slow one."
    )
    if not st.checkbox("Connect to the Data API", key="w_haz_api_on"):
        return

    try:
        client = datasets.api_client()
    except RuntimeError as err:
        components.error_box(err, "reach the Data API")
        return

    data_type = st.selectbox(
        "Data type", datasets.api_hazard_types(), key="w_haz_api_type"
    )
    try:
        with st.spinner("Listing available properties..."):
            properties = datasets.api_property_values(client, data_type)
    except Exception as err:
        components.error_box(err, "list datasets")
        return

    if not properties:
        st.warning(f"The API returned no active datasets for '{data_type}'.")
        return

    chosen = _property_picker(properties, key_prefix="w_haz_api")
    try:
        frame = datasets.api_dataset_frame(client, data_type, chosen)
    except Exception as err:
        components.error_box(err, "query the Data API")
        return

    if frame.empty:
        st.warning("No dataset matches that combination of properties.")
        return

    st.caption(f"{len(frame)} matching dataset(s)")
    st.dataframe(frame.head(50), use_container_width=True, hide_index=True)

    if st.button("Download and load", type="primary", key="w_haz_api_load"):
        try:
            with st.spinner("Downloading -- this can take a few minutes..."):
                hazard = client.get_hazard(data_type, properties=chosen or None)
        except Exception as err:
            components.error_box(err, "download the hazard")
            return
        state.put("hazard", hazard, invalidates=True)
        label = f"Data API: {data_type}"
        if chosen:
            label += " (" + ", ".join(f"{k}={v}" for k, v in chosen.items()) + ")"
        state.put("hazard_label", label)
        st.success(f"Loaded {hazard.size:,} {hazard.haz_type} events.")
        st.rerun()


def _property_picker(properties, key_prefix: str) -> dict:
    """Dropdowns for Data API dataset properties; returns the chosen subset."""
    chosen = {}
    columns = st.columns(min(3, max(1, len(properties))))
    for index, (name, values) in enumerate(sorted(properties.items())):
        options = ["(any)"] + [str(value) for value in values]
        with columns[index % len(columns)]:
            picked = st.selectbox(
                name.replace("_", " "),
                options,
                key=f"{key_prefix}_{name}",
            )
        if picked != "(any)":
            chosen[name] = picked
    return chosen


# --------------------------------------------------------------------------- #
# Exposures
# --------------------------------------------------------------------------- #


def _exposures_section() -> None:
    """Exposure loading: entity workbook, plain file, table or LitPop."""
    exposures = state.get("exposures")
    if exposures is not None:
        _exposures_summary(exposures)

    entity_tab, file_tab, api_tab, table_tab = st.tabs(
        [
            "Entity workbook",
            "Exposure file",
            "LitPop from the Data API",
            "Paste a table",
        ]
    )

    with entity_tab:
        st.markdown(
            "A CLIMADA entity spreadsheet carries exposures, impact functions, "
            "measures and discount rates in one workbook -- the fastest way to "
            "bring a complete study in."
        )
        uploaded = st.file_uploader(
            "Entity workbook", type=["xls", "xlsx", "mat"], key="w_ent_upload"
        )
        keep = st.checkbox(
            "Import its measures and discount rate too", value=True, key="w_ent_meas"
        )
        if uploaded is not None and st.button("Load entity", key="w_ent_load"):
            try:
                path = components.save_upload(uploaded, UPLOAD_DIR)
                with st.spinner("Reading entity..."):
                    entity = datasets.load_entity_file(path)
            except Exception as err:
                components.error_box(err, "load the entity")
            else:
                state.adopt_entity(entity, uploaded.name, keep_measures=keep)
                st.success("Entity loaded.")
                st.rerun()

    with file_tab:
        uploaded = st.file_uploader(
            "Exposures file",
            type=["h5", "hdf5", "csv", "mat"],
            key="w_exp_upload",
        )
        st.caption(
            "HDF5 written by `Exposures.write_hdf5`, or a CSV with `latitude`, "
            "`longitude` and `value` columns."
        )
        if uploaded is not None and st.button("Load exposures", key="w_exp_load"):
            try:
                path = components.save_upload(uploaded, UPLOAD_DIR)
                with st.spinner("Reading exposures..."):
                    loaded = datasets.load_exposures_file(path)
            except Exception as err:
                components.error_box(err, "load the exposures")
            else:
                state.put("exposures", loaded, invalidates=True)
                state.put("exposures_label", uploaded.name)
                st.success(f"Loaded {len(loaded.gdf):,} exposure points.")
                st.rerun()

    with api_tab:
        _api_litpop()

    with table_tab:
        _table_exposures()


def _exposures_summary(exposures) -> None:
    """Stat tiles and a map for the loaded exposures."""
    total = float(np.nansum(exposures.value)) if exposures.value is not None else 0.0
    components.stat_row(
        [
            ("Points", f"{len(exposures.gdf):,}", None),
            (
                "Total value",
                fmt_compact(total, exposures.value_unit),
                "Sum of the value column",
            ),
            ("Reference year", str(exposures.ref_year), None),
            (
                "Impact function columns",
                ", ".join(
                    col for col in exposures.gdf.columns if col.startswith("impf_")
                )
                or "none",
                "Which vulnerability curve each asset is assigned to",
            ),
        ]
    )
    st.caption(f"Source: {state.get('exposures_label') or 'unknown'}")

    frame = pd.DataFrame(
        {
            "latitude": exposures.latitude,
            "longitude": exposures.longitude,
            "value": exposures.value,
        }
    ).sort_values("value", ascending=False, ignore_index=True)
    limit = int(state.get("map_point_limit"))
    shown = frame.head(limit)
    figure = charts.point_map(
        shown,
        "value",
        unit=exposures.value_unit,
        dark=state.dark_mode(),
        title="Exposed value",
        basemap=bool(state.get("basemap")),
    )
    if len(frame) > limit:
        st.caption(f"Showing the {limit:,} highest-value of {len(frame):,} points.")
    components.chart_with_table(
        figure,
        shown.head(200),
        table_label="Show the highest-value points",
        download_name="exposures.csv",
        key="exposure-map",
    )


def _api_litpop() -> None:
    """Download a LitPop exposure layer for one country."""
    st.markdown(
        "LitPop disaggregates a country's produced capital onto a 150 arcsec "
        "grid using nightlights and population. It is CLIMADA's default "
        "global asset layer."
    )
    if not st.checkbox("Connect to the Data API", key="w_exp_api_on"):
        return

    try:
        client = datasets.api_client()
    except RuntimeError as err:
        components.error_box(err, "reach the Data API")
        return

    try:
        with st.spinner("Listing available countries..."):
            properties = datasets.api_property_values(client, "litpop")
    except Exception as err:
        components.error_box(err, "list LitPop datasets")
        return

    countries = sorted(str(value) for value in properties.get("country_name", []))
    if not countries:
        st.warning("The API returned no LitPop datasets.")
        return

    country = st.selectbox("Country", countries, key="w_exp_api_country")
    layer = st.radio(
        "Layer",
        ["Produced capital (USD)", "Population (people)"],
        help="Heat metrics count people, so heat analyses need the population "
        "layer. Damage-in-currency analyses need produced capital.",
        key="w_exp_api_layer",
        horizontal=True,
    )
    people = layer.startswith("Population")

    if st.button("Download LitPop", type="primary", key="w_exp_api_load"):
        try:
            with st.spinner(f"Downloading LitPop for {country}..."):
                exposures = (
                    datasets.api_population(client, country)
                    if people
                    else client.get_litpop(country)
                )
        except Exception as err:
            components.error_box(err, "download LitPop")
            return
        state.put("exposures", exposures, invalidates=True)
        state.put(
            "exposures_label",
            f"LitPop {country} ({'population' if people else 'produced capital'},"
            " Data API)",
        )
        st.success(f"Loaded {len(exposures.gdf):,} exposure points.")
        st.rerun()


def _table_exposures() -> None:
    """Type or paste a handful of exposure points directly."""
    st.markdown(
        "For a quick screening of a few sites, enter them here. Values are in "
        "the unit you name below."
    )
    seed = pd.DataFrame(
        {
            "latitude": [26.0, 25.8, 27.9],
            "longitude": [-80.2, -80.3, -82.5],
            "value": [5.0e8, 2.5e8, 1.2e9],
            "region_id": [1, 1, 2],
        }
    )
    edited = st.data_editor(
        seed,
        num_rows="dynamic",
        use_container_width=True,
        key="w_exp_table",
        column_config={
            "latitude": st.column_config.NumberColumn(format="%.4f"),
            "longitude": st.column_config.NumberColumn(format="%.4f"),
            "value": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    left, right = st.columns(2)
    unit = left.text_input("Value unit", value="USD", key="w_exp_table_unit")
    ref_year = right.number_input(
        "Reference year",
        min_value=1900,
        max_value=2200,
        value=int(state.get("present_year")),
        step=1,
        key="w_exp_table_year",
    )

    if st.button("Use this table", key="w_exp_table_load"):
        try:
            exposures = datasets.exposures_from_table(
                edited.dropna(subset=["latitude", "longitude", "value"]),
                value_unit=unit,
                ref_year=int(ref_year),
                haz_type=state.haz_type() or None,
            )
        except Exception as err:
            components.error_box(err, "build exposures from the table")
            return
        state.put("exposures", exposures, invalidates=True)
        state.put("exposures_label", "typed table")
        state.put("present_year", int(ref_year))
        st.success(f"Loaded {len(exposures.gdf):,} exposure points.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Vulnerability
# --------------------------------------------------------------------------- #


def _impf_section() -> None:
    """Impact function selection and the exposure-to-curve assignment."""
    impf_set = state.get("impf_set")
    haz_type = state.haz_type()

    if impf_set is not None:
        _impf_summary(impf_set, haz_type)

    heat_tab, default_tab, emanuel_tab, step_tab, file_tab = st.tabs(
        [
            "Heat",
            "Calibrated default",
            "Tropical cyclone (Emanuel)",
            "Step function",
            "From a file",
        ]
    )

    with heat_tab:
        _heat_impf(haz_type)

    with default_tab:
        if not haz_type:
            st.info("Load a hazard first -- the default depends on its type.")
        else:
            st.markdown(f"Load CLIMADA's shipped curves for **{haz_type}**.")
            if st.button("Load default curves", key="w_impf_default"):
                loaded, note = datasets.default_impf_set(haz_type)
                state.put("impf_set", loaded, invalidates=True)
                state.put("impf_label", f"CLIMADA default for {haz_type}")
                state.put("impf_note", note)
                st.rerun()

    with emanuel_tab:
        st.markdown(
            "The Emanuel (2011) curve: no damage below a threshold wind speed, "
            "then a cubic rise to saturation."
        )
        left, middle, right = st.columns(3)
        v_thresh = left.number_input(
            "No-damage threshold (m/s)",
            min_value=0.0,
            max_value=100.0,
            value=25.7,
            step=0.1,
            key="w_impf_vthresh",
        )
        v_half = middle.number_input(
            "Half-damage wind speed (m/s)",
            min_value=0.1,
            max_value=200.0,
            value=74.7,
            step=0.1,
            key="w_impf_vhalf",
        )
        scale = right.number_input(
            "Maximum damage ratio",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="w_impf_scale",
        )
        impf_id = st.number_input(
            "Impact function id", min_value=1, value=1, step=1, key="w_impf_eid"
        )
        if st.button("Use this curve", key="w_impf_emanuel"):
            try:
                loaded = datasets.emanuel_impf_set(
                    haz_type=haz_type or "TC",
                    impf_id=int(impf_id),
                    v_thresh=float(v_thresh),
                    v_half=float(v_half),
                    scale=float(scale),
                )
            except Exception as err:
                components.error_box(err, "build the Emanuel curve")
            else:
                state.put("impf_set", loaded, invalidates=True)
                state.put(
                    "impf_label",
                    f"Emanuel 2011 (v_thresh={v_thresh}, v_half={v_half})",
                )
                state.put("impf_note", "")
                st.rerun()

    with step_tab:
        st.markdown(
            "A blunt instrument for hazards with no calibrated curve: nothing "
            "below the threshold, a fixed damage ratio above it."
        )
        left, middle, right = st.columns(3)
        threshold = left.number_input(
            "Damage threshold", value=50.0, step=1.0, key="w_impf_step_thresh"
        )
        maximum = middle.number_input(
            "Axis maximum", value=100.0, step=1.0, key="w_impf_step_max"
        )
        ratio = right.number_input(
            "Damage ratio above threshold",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="w_impf_step_ratio",
        )
        if st.button("Use this curve", key="w_impf_step"):
            try:
                loaded = datasets.step_impf_set(
                    haz_type=haz_type or "HAZ",
                    intensity_high=float(threshold),
                    intensity_max=float(maximum),
                    damage_ratio=float(ratio),
                    intensity_unit=(
                        state.get("hazard").units if state.get("hazard") else ""
                    ),
                )
            except Exception as err:
                components.error_box(err, "build the step curve")
            else:
                state.put("impf_set", loaded, invalidates=True)
                state.put("impf_label", f"Step function above {threshold}")
                state.put("impf_note", "")
                st.rerun()

    with file_tab:
        uploaded = st.file_uploader(
            "Impact function workbook", type=["xls", "xlsx"], key="w_impf_upload"
        )
        if uploaded is not None and st.button("Load curves", key="w_impf_load"):
            try:
                path = components.save_upload(uploaded, UPLOAD_DIR)
                loaded = datasets.load_impf_file(path)
            except Exception as err:
                components.error_box(err, "load the impact functions")
            else:
                state.put("impf_set", loaded, invalidates=True)
                state.put("impf_label", uploaded.name)
                state.put("impf_note", "")
                st.rerun()

    if impf_set is not None and haz_type:
        _impf_assignment(impf_set, haz_type)


def _heat_impf(haz_type: str) -> None:
    """Build a temperature-response curve and record what it counts."""
    st.markdown(
        "CLIMADA ships no calibrated heat curve, so pick what you want to "
        "count and set the parameters from local evidence."
    )
    if haz_type and haz_type != heat.HAZ_TYPE:
        st.warning(
            f"The loaded hazard is '{haz_type}', not '{heat.HAZ_TYPE}'. A heat "
            "curve built here would not be found by the impact calculation."
        )

    metric = st.selectbox(
        "What to count",
        list(heat.METRICS),
        format_func=lambda key: heat.METRICS[key].label,
        key="w_heat_metric",
    )
    spec = heat.METRICS[metric]
    st.caption(f"{spec.description} Exposures must be in **{spec.exposure_unit}**.")

    kwargs = {}
    if metric == "mortality":
        cols = st.columns(3)
        kwargs["mmt"] = cols[0].number_input(
            "Minimum-mortality temperature (degC)",
            min_value=0.0,
            max_value=45.0,
            value=22.0,
            step=0.5,
            help="Below this, no excess deaths are attributed. Location- "
            "specific: roughly 18-20 degC in northern Europe, 26-29 in the "
            "tropics.",
            key="w_heat_mmt",
        )
        kwargs["rr_per_degree"] = cols[1].number_input(
            "Extra mortality per degree above it",
            min_value=0.0,
            max_value=0.30,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="Meta-analyses put all-age all-cause risk at 1-5% per degC, "
            "higher for the over-75s.",
            key="w_heat_rr",
        )
        kwargs["baseline_daily_mortality"] = cols[2].number_input(
            "Baseline deaths per person per day",
            min_value=0.0,
            max_value=1e-3,
            value=2.4e-5,
            step=1e-6,
            format="%.6f",
            help="2.4e-5 is about 8.8 deaths per 1,000 people per year. Raise "
            "it for an older population.",
            key="w_heat_baseline",
        )
    elif metric == "days_above":
        kwargs["threshold"] = st.number_input(
            "Threshold (degC)", value=35.0, step=0.5, key="w_heat_days_thresh"
        )
    elif metric == "degree_days":
        kwargs["threshold"] = st.number_input(
            "Base temperature (degC)", value=30.0, step=0.5, key="w_heat_dd_thresh"
        )
    elif metric == "labour":
        cols = st.columns(2)
        kwargs["work_start"] = cols[0].number_input(
            "Productivity starts falling (degC)",
            value=26.0,
            step=0.5,
            key="w_heat_wstart",
        )
        kwargs["work_stop"] = cols[1].number_input(
            "Work stops entirely (degC)",
            value=38.0,
            step=0.5,
            key="w_heat_wstop",
        )

    if not st.button("Use this curve", type="primary", key="w_heat_build"):
        return

    try:
        impf_set = heat.impf_set_for_metric(metric, **kwargs)
    except (ValueError, KeyError) as err:
        st.error(str(err))
        return

    state.put("impf_set", impf_set, invalidates=True)
    state.put("impf_label", spec.label)
    state.put(
        "impf_note",
        f"Counting {spec.unit}. {spec.description} This is a screening "
        "relationship, not an epidemiological study -- calibrate it before "
        "reporting the numbers.",
    )
    state.put("heat_metric", metric)
    st.rerun()


def _impf_summary(impf_set, haz_type: str) -> None:
    """Plot the loaded vulnerability curves."""
    note = state.get("impf_note")
    if note:
        st.info(note)

    curves = datasets.impf_curves(impf_set, haz_type or None)
    if not curves:
        st.warning(
            f"The loaded set has no curve for hazard type '{haz_type}'. "
            "The risk calculation will fail until one is added."
        )
        return

    shown = dict(list(curves.items())[: charts.MAX_CATEGORICAL_SERIES])
    if len(curves) > len(shown):
        st.caption(
            f"Showing {len(shown)} of {len(curves)} curves -- the chart holds "
            f"{charts.MAX_CATEGORICAL_SERIES} distinguishable series."
        )

    figure = charts.impact_function_chart(
        shown,
        intensity_unit=datasets.impf_intensity_unit(impf_set, haz_type or None),
        dark=state.dark_mode(),
    )
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "Curve": label,
                    "Intensity": curve["intensity"],
                    "Mean damage ratio": curve["mdr"],
                }
            )
            for label, curve in shown.items()
        ],
        ignore_index=True,
    )
    components.chart_with_table(
        figure,
        frame,
        table_label="Show the curve values",
        download_name="impact_functions.csv",
        key="impf-chart",
    )
    st.caption(f"Source: {state.get('impf_label') or 'unknown'}")


def _impf_assignment(impf_set, haz_type: str) -> None:
    """Let the user point all exposures at one impact function id."""
    exposures = state.get("exposures")
    if exposures is None:
        return

    available = datasets.impf_ids(impf_set, haz_type)
    if not available:
        return

    column = f"impf_{haz_type}"
    assigned: List[int] = []
    if column in exposures.gdf.columns:
        assigned = sorted(int(value) for value in exposures.gdf[column].unique())

    unmatched = [value for value in assigned if value not in available]
    if not assigned:
        st.warning(
            f"The exposures have no `{column}` column, so no asset knows which "
            "curve applies to it."
        )
    elif unmatched:
        st.warning(
            f"Exposures reference impact function id(s) {unmatched} that the "
            f"loaded set does not define (it has {available})."
        )

    if assigned and not unmatched:
        return

    with st.form("w_impf_assign_form"):
        st.markdown("**Assign every exposure point to one curve**")
        chosen = st.selectbox("Impact function id", available, key="w_impf_assign_id")
        submitted = st.form_submit_button("Assign")
    if submitted:
        state.put(
            "exposures",
            datasets.assign_impf_id(exposures, haz_type, int(chosen)),
            invalidates=True,
        )
        st.success(f"All exposure points now use impact function {chosen}.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


def _readiness_section() -> None:
    """Report what is loaded and whether the pieces fit together."""
    st.subheader("What is loaded")
    summary = state.state_summary()
    for name, description in summary.items():
        if description:
            st.markdown(f"- **{name.capitalize()}**: {description}")
        else:
            st.markdown(f"- **{name.capitalize()}**: nothing loaded")

    problems = _compatibility_problems()
    if problems:
        for problem in problems:
            st.warning(problem)
    elif state.inputs_ready():
        st.success("Ready to run. Open **Risk** for the physical risk assessment.")


def _compatibility_problems() -> List[str]:
    """Checks that catch the mismatches CLIMADA would otherwise raise on."""
    problems: List[str] = []
    hazard = state.get("hazard")
    exposures = state.get("exposures")
    impf_set = state.get("impf_set")
    if hazard is None or exposures is None or impf_set is None:
        return problems

    haz_type = hazard.haz_type
    column = f"impf_{haz_type}"
    columns = list(exposures.gdf.columns)
    if column not in columns and "impf_" not in columns:
        problems.append(
            f"The exposures carry no `{column}` column, so CLIMADA cannot tell "
            f"which vulnerability curve applies to each asset for hazard "
            f"'{haz_type}'. Assign one on the Vulnerability tab."
        )

    if not impf_set.get_func(haz_type=haz_type):
        problems.append(
            f"The impact function set defines no curve for hazard '{haz_type}'."
        )

    haz_bounds = hazard.centroids.total_bounds
    exp_lon = np.asarray(exposures.longitude, dtype=float)
    exp_lat = np.asarray(exposures.latitude, dtype=float)
    if exp_lon.size and (
        exp_lon.max() < haz_bounds[0]
        or exp_lon.min() > haz_bounds[2]
        or exp_lat.max() < haz_bounds[1]
        or exp_lat.min() > haz_bounds[3]
    ):
        problems.append(
            "The exposures lie entirely outside the hazard footprint, so every "
            "impact would be zero. Check that the two describe the same region."
        )

    return problems
