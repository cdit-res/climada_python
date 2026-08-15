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

Loading hazards, exposures and impact functions for the CLIMADA user interface.

Three sources are supported: the demo data bundled with CLIMADA, files on disk
or uploaded by the user, and the CLIMADA Data API. Nothing here imports
Streamlit.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from climada.entity import Entity, Exposures, ImpactFunc, ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone, ImpfTropCyclone
from climada.hazard import Hazard
from climada.util.api_client import EXP_TYPES, HAZ_TYPES, Client
from climada.util.constants import (
    ENT_DEMO_FUTURE,
    ENT_DEMO_TODAY,
    ENT_TEMPLATE_XLS,
    HAZ_DEMO_H5,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEMO_SCENARIOS",
    "DemoScenario",
    "api_client",
    "api_dataset_frame",
    "api_hazard_types",
    "api_property_values",
    "default_impf_set",
    "exposures_from_table",
    "impf_curves",
    "load_demo",
    "load_entity_file",
    "load_exposures_file",
    "load_hazard_file",
    "load_impf_file",
    "step_impf_set",
]

HAZARD_SUFFIXES = (".h5", ".hdf5", ".xls", ".xlsx")
EXPOSURES_SUFFIXES = (".h5", ".hdf5", ".xls", ".xlsx", ".csv", ".mat")


# --------------------------------------------------------------------------- #
# Bundled demo data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DemoScenario:
    """A ready-made analysis shipped with CLIMADA.

    Attributes
    ----------
    key : str
        Stable identifier.
    label : str
        Name shown in the interface.
    description : str
        One-line summary of what the data is.
    hazard_path : pathlib.Path
        HDF5 hazard event set.
    entity_path : pathlib.Path
        Excel entity holding exposures, impact functions, measures and rates.
    future_entity_path : pathlib.Path, optional
        Entity describing the future state, if the demo has one.
    """

    key: str
    label: str
    description: str
    hazard_path: Path
    entity_path: Path
    future_entity_path: Optional[Path] = None

    @property
    def available(self) -> bool:
        """Whether every file this scenario needs is present on disk."""
        paths = [self.hazard_path, self.entity_path]
        if self.future_entity_path is not None:
            paths.append(self.future_entity_path)
        return all(path.is_file() for path in paths)


DEMO_SCENARIOS: Dict[str, DemoScenario] = {
    "tc_florida": DemoScenario(
        key="tc_florida",
        label="Tropical cyclone, Florida",
        description=(
            "CLIMADA's demo set: 216 Atlantic tropical cyclone events over "
            "1990-2004 against Florida coastal assets, with four coastal "
            "adaptation measures already specified."
        ),
        hazard_path=Path(HAZ_DEMO_H5),
        entity_path=Path(ENT_DEMO_TODAY),
        future_entity_path=Path(ENT_DEMO_FUTURE),
    ),
    "entity_template": DemoScenario(
        key="entity_template",
        label="Entity template (blank starting point)",
        description=(
            "CLIMADA's entity spreadsheet template, paired with the demo "
            "tropical cyclone hazard. Use it as a skeleton for your own data."
        ),
        hazard_path=Path(HAZ_DEMO_H5),
        entity_path=Path(ENT_TEMPLATE_XLS),
    ),
}
"""Demo analyses selectable in the interface, keyed by identifier."""


def load_demo(key: str) -> Dict[str, Any]:
    """Load a bundled demo scenario.

    Parameters
    ----------
    key : str
        Key into :py:data:`DEMO_SCENARIOS`.

    Returns
    -------
    dict
        ``hazard``, ``entity``, ``entity_future`` (possibly ``None``) and
        ``scenario``.

    Raises
    ------
    KeyError
        If ``key`` is not a known demo.
    FileNotFoundError
        If the demo's files are missing from the installation.
    """
    scenario = DEMO_SCENARIOS[key]
    if not scenario.available:
        raise FileNotFoundError(
            f"Demo '{scenario.label}' is missing its data files. Expected "
            f"{scenario.hazard_path} and {scenario.entity_path}."
        )

    hazard = Hazard.from_hdf5(scenario.hazard_path)
    entity = Entity.from_excel(scenario.entity_path)
    entity.check()

    entity_future = None
    if scenario.future_entity_path is not None:
        entity_future = Entity.from_excel(scenario.future_entity_path)
        entity_future.check()

    return {
        "hazard": hazard,
        "entity": entity,
        "entity_future": entity_future,
        "scenario": scenario,
    }


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def load_hazard_file(path: Path, haz_type: Optional[str] = None) -> Hazard:
    """Read a hazard from an HDF5 or Excel file.

    Parameters
    ----------
    path : pathlib.Path
        File to read. The suffix selects the reader.
    haz_type : str, optional
        Hazard type to stamp on the result, for formats that do not carry one.

    Returns
    -------
    climada.hazard.Hazard

    Raises
    ------
    ValueError
        If the suffix is not one CLIMADA can read as a hazard.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        hazard = Hazard.from_hdf5(path)
    elif suffix in (".xls", ".xlsx"):
        hazard = Hazard.from_excel(path, haz_type=haz_type)
    else:
        raise ValueError(
            f"Cannot read '{path.name}' as a hazard. Supported: "
            f"{', '.join(HAZARD_SUFFIXES)}."
        )
    if haz_type and not hazard.haz_type:
        hazard.haz_type = haz_type
    hazard.check()
    return hazard


def load_exposures_file(path: Path) -> Exposures:
    """Read exposures from HDF5, MATLAB, Excel or CSV.

    Excel files are read as full CLIMADA entities and their exposures returned.
    CSV files must carry ``latitude``, ``longitude`` and ``value`` columns.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    climada.entity.Exposures

    Raises
    ------
    ValueError
        If the suffix is unsupported, or a CSV lacks the required columns.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        exposures = Exposures.from_hdf5(path)
    elif suffix == ".mat":
        exposures = Exposures.from_mat(path)
    elif suffix in (".xls", ".xlsx"):
        exposures = Entity.from_excel(path).exposures
    elif suffix == ".csv":
        exposures = exposures_from_table(pd.read_csv(path))
    else:
        raise ValueError(
            f"Cannot read '{path.name}' as exposures. Supported: "
            f"{', '.join(EXPOSURES_SUFFIXES)}."
        )
    exposures.check()
    return exposures


def exposures_from_table(
    frame: pd.DataFrame,
    value_unit: str = "USD",
    ref_year: Optional[int] = None,
    haz_type: Optional[str] = None,
    impf_id: int = 1,
) -> Exposures:
    """Build exposures from a plain table of points.

    Parameters
    ----------
    frame : pandas.DataFrame
        Needs ``latitude``/``lat``, ``longitude``/``lon`` and ``value``.
        Optional extra columns (``region_id``, ``category_id``, ``deductible``,
        ``cover``, ``impf_*``) are carried through.
    value_unit : str, optional
        Unit of the ``value`` column. Default: ``'USD'``.
    ref_year : int, optional
        Reference year of the asset values.
    haz_type : str, optional
        When given and no impact function column is present, add ``impf_<type>``.
    impf_id : int, optional
        Impact function id written into that column. Default: 1.

    Returns
    -------
    climada.entity.Exposures

    Raises
    ------
    ValueError
        If the coordinate or value columns are missing.
    """
    frame = frame.rename(
        columns={"lat": "latitude", "lon": "longitude", "Value": "value"}
    )
    missing = {"latitude", "longitude", "value"}.difference(frame.columns)
    if missing:
        raise ValueError(
            "Exposure table needs columns latitude, longitude and value; "
            f"missing {', '.join(sorted(missing))}."
        )

    exposures = Exposures(
        frame,
        value_unit=value_unit,
        ref_year=ref_year,
    )
    if haz_type:
        column = f"impf_{haz_type}"
        has_impf = any(col.startswith("impf_") for col in exposures.gdf.columns)
        if not has_impf:
            exposures.gdf[column] = int(impf_id)
    exposures.check()
    return exposures


def load_impf_file(path: Path) -> ImpactFuncSet:
    """Read an impact function set from an Excel workbook.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    climada.entity.ImpactFuncSet

    Raises
    ------
    ValueError
        If the suffix is not an Excel one.
    """
    path = Path(path)
    if path.suffix.lower() not in (".xls", ".xlsx"):
        raise ValueError(
            f"Cannot read '{path.name}' as impact functions. Supported: .xls, .xlsx."
        )
    impf_set = ImpactFuncSet.from_excel(path)
    impf_set.check()
    return impf_set


def load_entity_file(path: Path) -> Entity:
    """Read a full entity from an Excel workbook or MATLAB file.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    climada.entity.Entity

    Raises
    ------
    ValueError
        If the suffix is unsupported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        entity = Entity.from_excel(path)
    elif suffix == ".mat":
        entity = Entity.from_mat(path)
    else:
        raise ValueError(
            f"Cannot read '{path.name}' as an entity. Supported: .xls, .xlsx, .mat."
        )
    entity.check()
    return entity


# --------------------------------------------------------------------------- #
# Impact functions
# --------------------------------------------------------------------------- #


def default_impf_set(haz_type: str) -> Tuple[ImpactFuncSet, str]:
    """A sensible starting vulnerability set for a hazard type.

    Parameters
    ----------
    haz_type : str
        CLIMADA hazard type, e.g. ``'TC'``.

    Returns
    -------
    tuple of (climada.entity.ImpactFuncSet, str)
        The set and a short note naming its provenance.
    """
    if haz_type == "TC":
        return (
            ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(),
            "Emanuel (2011) tropical cyclone curves, regionally calibrated "
            "(Eberenz et al. 2021). Impact function ids follow CLIMADA's region "
            "codes; assign the matching id to your exposures.",
        )

    impf_set = ImpactFuncSet(
        [
            ImpactFunc.from_step_impf(
                intensity=(0.0, 1.0, 100.0),
                haz_type=haz_type or "HAZ",
                impf_id=1,
                intensity_unit="",
            )
        ]
    )
    impf_set.check()
    return (
        impf_set,
        f"No calibrated default ships with CLIMADA for '{haz_type}'. A placeholder "
        "step function is loaded -- replace it with your own curve before "
        "drawing conclusions.",
    )


def emanuel_impf_set(
    haz_type: str = "TC",
    impf_id: int = 1,
    v_thresh: float = 25.7,
    v_half: float = 74.7,
    scale: float = 1.0,
) -> ImpactFuncSet:
    """A single Emanuel-type tropical cyclone curve with editable parameters.

    Parameters
    ----------
    haz_type : str, optional
        Hazard type to register the function under. Default: ``'TC'``.
    impf_id : int, optional
        Function id. Default: 1.
    v_thresh : float, optional
        Wind speed in m/s below which no damage occurs. Default: 25.7.
    v_half : float, optional
        Wind speed in m/s at which half the maximum damage is reached. Must
        exceed ``v_thresh``. Default: 74.7.
    scale : float, optional
        Maximum damage ratio, in (0, 1]. Default: 1.0.

    Returns
    -------
    climada.entity.ImpactFuncSet
    """
    impf = ImpfTropCyclone.from_emanuel_usa(
        impf_id=impf_id, v_thresh=v_thresh, v_half=v_half, scale=scale
    )
    impf.haz_type = haz_type
    impf_set = ImpactFuncSet([impf])
    impf_set.check()
    return impf_set


def step_impf_set(
    haz_type: str,
    impf_id: int = 1,
    intensity_low: float = 0.0,
    intensity_high: float = 50.0,
    intensity_max: float = 100.0,
    damage_ratio: float = 1.0,
    intensity_unit: str = "",
) -> ImpactFuncSet:
    """A step vulnerability curve, for hazards with no calibrated default.

    Parameters
    ----------
    haz_type : str
        Hazard type to register the function under.
    impf_id : int, optional
        Function id. Default: 1.
    intensity_low : float, optional
        Intensity below which nothing is damaged. Default: 0.
    intensity_high : float, optional
        Intensity above which ``damage_ratio`` applies. Default: 50.
    intensity_max : float, optional
        Upper end of the intensity axis. Default: 100.
    damage_ratio : float, optional
        Damage ratio above the step, in [0, 1]. Default: 1.0.
    intensity_unit : str, optional
        Unit shown on charts.

    Returns
    -------
    climada.entity.ImpactFuncSet
    """
    impf = ImpactFunc.from_step_impf(
        intensity=(intensity_low, intensity_high, intensity_max),
        haz_type=haz_type,
        impf_id=impf_id,
        intensity_unit=intensity_unit,
        mdd=(0.0, float(damage_ratio)),
    )
    impf_set = ImpactFuncSet([impf])
    impf_set.check()
    return impf_set


def impf_curves(
    impf_set: ImpactFuncSet, haz_type: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Extract plottable mean-damage-ratio curves from an impact function set.

    Parameters
    ----------
    impf_set : climada.entity.ImpactFuncSet
    haz_type : str, optional
        Restrict to one hazard type. Default: all of them.

    Returns
    -------
    dict
        Mapping of label to ``{'intensity': array, 'mdr': array}``, where the
        mean damage ratio is ``mdd * paa``.
    """
    if haz_type is None:
        functions = [
            impf for by_id in impf_set.get_func().values() for impf in by_id.values()
        ]
    else:
        functions = impf_set.get_func(haz_type=haz_type)

    curves: Dict[str, Dict[str, Any]] = {}
    for impf in functions:
        label = f"{impf.haz_type} #{impf.id}"
        if impf.name and impf.name != str(impf.id):
            label = f"{label} - {impf.name}"
        curves[label] = {
            "intensity": np.asarray(impf.intensity, dtype=float),
            "mdr": np.asarray(impf.mdd, dtype=float)
            * np.asarray(impf.paa, dtype=float),
            "unit": impf.intensity_unit,
        }
    return curves


def impf_intensity_unit(impf_set: ImpactFuncSet, haz_type: Optional[str] = None) -> str:
    """The intensity unit of the first function in a set, or an empty string.

    Parameters
    ----------
    impf_set : climada.entity.ImpactFuncSet
    haz_type : str, optional

    Returns
    -------
    str
    """
    curves = impf_curves(impf_set, haz_type)
    for curve in curves.values():
        if curve["unit"]:
            return curve["unit"]
    return ""


def impf_ids(impf_set: ImpactFuncSet, haz_type: str) -> List[int]:
    """Impact function ids defined for a hazard type.

    Parameters
    ----------
    impf_set : climada.entity.ImpactFuncSet
    haz_type : str

    Returns
    -------
    list of int
    """
    return sorted(int(fid) for fid in impf_set.get_ids(haz_type=haz_type))


def assign_impf_id(exposures: Exposures, haz_type: str, impf_id: int) -> Exposures:
    """Point every exposure point at one impact function id.

    Parameters
    ----------
    exposures : climada.entity.Exposures
        Left untouched; a copy is returned.
    haz_type : str
        Hazard type, giving the ``impf_<type>`` column name.
    impf_id : int
        Id to write.

    Returns
    -------
    climada.entity.Exposures
    """
    updated = exposures.copy(deep=True)
    updated.gdf[f"impf_{haz_type}"] = int(impf_id)
    return updated


# --------------------------------------------------------------------------- #
# CLIMADA Data API
# --------------------------------------------------------------------------- #


def api_client() -> Client:
    """A CLIMADA Data API client.

    Returns
    -------
    climada.util.api_client.Client

    Raises
    ------
    RuntimeError
        If the API cannot be reached.
    """
    try:
        return Client()
    # The client constructor reaches the network; every failure mode there
    # means the same thing to the caller.
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "Could not reach the CLIMADA Data API. Check the network connection, "
            "or load data from a file instead."
        ) from err


def api_hazard_types() -> List[str]:
    """Hazard data types the Data API offers.

    Returns
    -------
    list of str
    """
    return sorted(HAZ_TYPES)


def api_exposures_types() -> List[str]:
    """Exposures data types the Data API offers.

    Returns
    -------
    list of str
    """
    return sorted(EXP_TYPES)


def api_property_values(
    client: Client, data_type: str, known: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    """Selectable property values for a Data API data type.

    Parameters
    ----------
    client : climada.util.api_client.Client
    data_type : str
        e.g. ``'tropical_cyclone'`` or ``'litpop'``.
    known : dict, optional
        Property values already chosen, to narrow the remaining options.

    Returns
    -------
    dict
        Mapping of property name to the values still available.
    """
    infos = client.list_dataset_infos(data_type=data_type)
    if not infos:
        return {}
    return client.get_property_values(infos, known_property_values=known or None)


def api_dataset_frame(
    client: Client, data_type: str, properties: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Datasets matching a query, as a browsable table.

    Parameters
    ----------
    client : climada.util.api_client.Client
    data_type : str
    properties : dict, optional
        Property filters.

    Returns
    -------
    pandas.DataFrame
        Empty when nothing matches.
    """
    infos = client.list_dataset_infos(
        data_type=data_type, properties=properties or None
    )
    if not infos:
        return pd.DataFrame()
    return client.into_datasets_df(infos)
