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

Heat risk for the CLIMADA user interface.

CLIMADA's core ships no heat hazard class and no heat vulnerability curves --
the documentation names heat mortality as the archetypal use of the
proportion-of-assets-affected term, but leaves the curve to the user. This
module supplies the missing pieces:

* reading gridded temperature into a :py:class:`~climada.hazard.Hazard` of
  type ``HW``, with the event frequency that daily data actually implies;
* a small library of temperature-response functions -- excess mortality,
  person-days above a threshold, degree-days, and lost labour;
* population exposure helpers, and heat-specific adaptation measures.

Nothing here imports Streamlit.

Notes
-----
Following CLIMADA's own convention for heat (see the overview tutorial: "the
Proportion of Assets Affected (PAA) gives the fraction of exposures that are
affected, such as the mortality rate in a population from a heatwave"), the
temperature response is carried in ``paa`` and ``mdd`` is held at 1. Mean
damage ratio is the product either way, but the split decides what an
adaptation measure means: ``paa_impact`` then reads as "this measure cuts the
harm rate at a given temperature", which is how heat interventions are
normally described.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)

__all__ = [
    "HAZ_TYPE",
    "HEAT_MEASURE_PRESETS",
    "HeatMetric",
    "METRICS",
    "annual_frequency",
    "days_above_impf_set",
    "degree_days_impf_set",
    "demo_bundle",
    "demo_heat_hazard",
    "demo_population",
    "hazard_from_dataset",
    "hazard_temperature_summary",
    "impact_unit",
    "metric_for",
    "labour_loss_impf_set",
    "mortality_impf_set",
    "set_annual_frequency",
]


HAZ_TYPE = "HW"
"""CLIMADA's hazard-type code for extreme temperature.

Matches the code used for EM-DAT's "Extreme temperature" peril in
:py:mod:`climada.engine.impact_data`, so heat impacts computed here line up
with recorded heat disasters.
"""

DEFAULT_INTENSITY_UNIT = "degC"


# --------------------------------------------------------------------------- #
# What is being counted
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeatMetric:
    """What a heat impact number means, so the interface can label it.

    A heat analysis can count deaths, person-days of exposure, degree-days or
    lost working hours. They share the CLIMADA machinery but read completely
    differently, and mislabelling them is the easiest way to mislead.

    Attributes
    ----------
    key : str
        Stable identifier.
    label : str
        Name of the metric, for menus.
    unit : str
        Unit of the resulting impact, e.g. ``'deaths'``.
    exposure_unit : str
        Unit the exposures must be in for the metric to make sense.
    annual_noun : str
        How to describe the average annual impact in prose.
    description : str
        One sentence on what the curve represents.
    per_capita_basis : float, optional
        Population base for a rate, e.g. 100000 for "per 100,000". ``None``
        when a rate makes no sense for this metric.
    """

    key: str
    label: str
    unit: str
    exposure_unit: str
    annual_noun: str
    description: str
    per_capita_basis: Optional[float] = None


METRICS: Dict[str, HeatMetric] = {
    "mortality": HeatMetric(
        key="mortality",
        label="Heat-attributable mortality",
        unit="deaths",
        exposure_unit="people",
        annual_noun="heat-attributable deaths per year",
        description=(
            "Excess deaths above the minimum-mortality temperature, from a "
            "baseline death rate and a relative risk that rises with heat."
        ),
        per_capita_basis=100_000.0,
    ),
    "days_above": HeatMetric(
        key="days_above",
        label="Person-days above a threshold",
        unit="person-days",
        exposure_unit="people",
        annual_noun="person-days of dangerous heat per year",
        description=(
            "Counts everyone exposed on every day at or above the threshold. "
            "No vulnerability assumption at all -- pure exposure."
        ),
    ),
    "degree_days": HeatMetric(
        key="degree_days",
        label="Person-degree-days",
        unit="person-degree-days",
        exposure_unit="people",
        annual_noun="person-degree-days per year",
        description=(
            "Person-days weighted by how far the temperature exceeds the "
            "threshold, so a 40 degC day counts for more than a 36 degC day."
        ),
    ),
    "labour": HeatMetric(
        key="labour",
        label="Lost labour",
        unit="lost work-days",
        exposure_unit="workers",
        annual_noun="work-days lost to heat per year",
        description=(
            "Fraction of the working day lost to mandated rest at high heat, "
            "after the ISO 7243 work-rest schedule."
        ),
    ),
}
"""The heat metrics the interface can compute, keyed by identifier."""


def metric_for(haz_type: str, key: Optional[str]) -> Optional[HeatMetric]:
    """The heat metric in play, or ``None`` when this is not a heat analysis.

    Guards the two ways a label can go stale: a metric recorded against a
    hazard that is no longer heat, and a key that no longer names a metric.

    Parameters
    ----------
    haz_type : str
        Hazard type of the analysis.
    key : str or None
        Metric key recorded when the vulnerability curve was built.

    Returns
    -------
    HeatMetric or None
    """
    if not key or haz_type != HAZ_TYPE:
        return None
    return METRICS.get(key)


def impact_unit(haz_type: str, key: Optional[str], fallback: str) -> str:
    """The unit the impact numbers are in, which is not the exposure unit.

    A heat mortality run has exposures in people but an impact in deaths;
    labelling the result "people" would misreport it.

    Parameters
    ----------
    haz_type : str
        Hazard type of the analysis.
    key : str or None
        Metric key recorded when the vulnerability curve was built.
    fallback : str
        Unit to use when this is not a heat analysis, normally the exposures'.

    Returns
    -------
    str
    """
    metric = metric_for(haz_type, key)
    return metric.unit if metric is not None else fallback


# --------------------------------------------------------------------------- #
# Vulnerability curves
# --------------------------------------------------------------------------- #


def mortality_impf_set(
    mmt: float = 22.0,
    rr_per_degree: float = 0.03,
    baseline_daily_mortality: float = 2.4e-5,
    max_temperature: float = 55.0,
    impf_id: int = 1,
    intensity_unit: str = DEFAULT_INTENSITY_UNIT,
    step: float = 0.5,
) -> ImpactFuncSet:
    """Excess heat mortality as a function of daily temperature.

    Implements the standard epidemiological form: no excess deaths at or below
    the minimum-mortality temperature (MMT), and above it a relative risk
    rising linearly with temperature. The per-person, per-day excess death
    rate at temperature ``T`` is

    ``baseline_daily_mortality * rr_per_degree * max(T - mmt, 0)``

    which is the baseline death rate multiplied by ``RR - 1``.

    The result is a *daily* rate, so the hazard must be a set of daily
    temperature fields whose frequencies are per year -- see
    :py:func:`set_annual_frequency`. The average annual impact is then
    expected heat-attributable deaths per year.

    Parameters
    ----------
    mmt : float, optional
        Minimum-mortality temperature, below which no excess deaths are
        attributed. Typically the 60th-90th percentile of the local
        temperature distribution, so it is location-specific: roughly 18-20
        degC in northern Europe, 26-29 degC in the tropics. Default: 22.
    rr_per_degree : float, optional
        Fractional increase in daily mortality per degree above the MMT.
        Meta-analyses put this in the 1-5% range for all-age all-cause
        mortality, higher for the over-75s. Default: 0.03.
    baseline_daily_mortality : float, optional
        Baseline all-cause deaths per person per day. The default 2.4e-5
        corresponds to about 8.8 deaths per 1,000 people per year, a typical
        high-income crude death rate. Raise it for an older population.
    max_temperature : float, optional
        Upper end of the curve's temperature axis. Default: 55.
    impf_id : int, optional
        Impact function id. Default: 1.
    intensity_unit : str, optional
        Unit shown on charts. Default: ``'degC'``.
    step : float, optional
        Temperature resolution of the tabulated curve. Default: 0.5.

    Returns
    -------
    climada.entity.ImpactFuncSet

    Raises
    ------
    ValueError
        If the parameters are outside a physically meaningful range.

    Notes
    -----
    This is a screening relationship, not an epidemiological study. Real
    exposure-response curves are non-linear, lagged over several days, and
    differ by age, city and acclimatisation. Calibrate ``mmt`` and
    ``rr_per_degree`` to local literature before reporting death counts.
    """
    if rr_per_degree < 0:
        raise ValueError("rr_per_degree must not be negative.")
    if baseline_daily_mortality < 0:
        raise ValueError("baseline_daily_mortality must not be negative.")
    if max_temperature <= mmt:
        raise ValueError(
            f"max_temperature ({max_temperature}) must exceed the MMT ({mmt})."
        )

    intensity = _axis(mmt, max_temperature, step)
    excess = baseline_daily_mortality * rr_per_degree * np.maximum(intensity - mmt, 0.0)
    return _build(
        intensity,
        paa=np.minimum(excess, 1.0),
        impf_id=impf_id,
        name=f"Heat mortality (MMT {mmt:g} degC, {rr_per_degree:.1%}/degC)",
        intensity_unit=intensity_unit,
    )


def days_above_impf_set(
    threshold: float = 35.0,
    max_temperature: float = 55.0,
    impf_id: int = 1,
    intensity_unit: str = DEFAULT_INTENSITY_UNIT,
) -> ImpactFuncSet:
    """Every exposed person counts on every day at or above a threshold.

    Makes no vulnerability assumption: the impact is the exposure itself,
    counted on hot days. With daily events at annual frequencies, the average
    annual impact is person-days of dangerous heat per year.

    Parameters
    ----------
    threshold : float, optional
        Temperature at and above which a day counts. Default: 35.
    max_temperature : float, optional
        Upper end of the temperature axis. Default: 55.
    impf_id : int, optional
        Impact function id. Default: 1.
    intensity_unit : str, optional

    Returns
    -------
    climada.entity.ImpactFuncSet

    Raises
    ------
    ValueError
        If the axis does not extend above the threshold.
    """
    if max_temperature <= threshold:
        raise ValueError(
            f"max_temperature ({max_temperature}) must exceed the threshold "
            f"({threshold})."
        )
    # A hair below the threshold keeps the step sharp under interpolation.
    intensity = np.array(
        [0.0, threshold - 1e-6, threshold, max_temperature], dtype=float
    )
    return _build(
        intensity,
        paa=np.array([0.0, 0.0, 1.0, 1.0]),
        impf_id=impf_id,
        name=f"Days at or above {threshold:g} degC",
        intensity_unit=intensity_unit,
    )


def degree_days_impf_set(
    threshold: float = 30.0,
    max_temperature: float = 55.0,
    impf_id: int = 1,
    intensity_unit: str = DEFAULT_INTENSITY_UNIT,
    step: float = 0.5,
) -> ImpactFuncSet:
    """Exposure weighted by how far the temperature exceeds a threshold.

    The impact per exposed person per day is ``max(T - threshold, 0)``, so the
    average annual impact is person-degree-days per year.

    Parameters
    ----------
    threshold : float, optional
        Base temperature of the degree-day count. Default: 30.
    max_temperature : float, optional
        Upper end of the temperature axis. Default: 55.
    impf_id : int, optional
        Impact function id. Default: 1.
    intensity_unit : str, optional
    step : float, optional
        Temperature resolution of the tabulated curve. Default: 0.5.

    Returns
    -------
    climada.entity.ImpactFuncSet

    Raises
    ------
    ValueError
        If the axis does not extend above the threshold.
    """
    if max_temperature <= threshold:
        raise ValueError(
            f"max_temperature ({max_temperature}) must exceed the threshold "
            f"({threshold})."
        )
    intensity = _axis(threshold, max_temperature, step)
    return _build(
        intensity,
        paa=np.maximum(intensity - threshold, 0.0),
        impf_id=impf_id,
        name=f"Degree-days above {threshold:g} degC",
        intensity_unit=intensity_unit,
    )


def labour_loss_impf_set(
    work_start: float = 26.0,
    work_stop: float = 38.0,
    impf_id: int = 1,
    intensity_unit: str = DEFAULT_INTENSITY_UNIT,
    step: float = 0.5,
) -> ImpactFuncSet:
    """Fraction of the working day lost to heat, rising between two bounds.

    A piecewise-linear stand-in for the ISO 7243 work-rest schedule: no rest
    required at or below ``work_start``, all work stopped at or above
    ``work_stop``, linear in between. With daily events at annual
    frequencies, the average annual impact is work-days lost per year.

    Parameters
    ----------
    work_start : float, optional
        Temperature at which productivity begins to fall. Default: 26.
    work_stop : float, optional
        Temperature at which work stops entirely. Default: 38.
    impf_id : int, optional
        Impact function id. Default: 1.
    intensity_unit : str, optional
    step : float, optional
        Temperature resolution of the tabulated curve. Default: 0.5.

    Returns
    -------
    climada.entity.ImpactFuncSet

    Raises
    ------
    ValueError
        If ``work_stop`` does not exceed ``work_start``.

    Notes
    -----
    Real schedules depend on wet-bulb globe temperature and on work
    intensity: heavy outdoor labour loses capacity far earlier than office
    work. Treat the two bounds as the knobs that encode the workforce.
    """
    if work_stop <= work_start:
        raise ValueError(
            f"work_stop ({work_stop}) must exceed work_start ({work_start})."
        )
    intensity = _axis(work_start, work_stop + 10.0, step)
    fraction = np.clip((intensity - work_start) / (work_stop - work_start), 0.0, 1.0)
    return _build(
        intensity,
        paa=fraction,
        impf_id=impf_id,
        name=f"Labour loss ({work_start:g}-{work_stop:g} degC)",
        intensity_unit=intensity_unit,
    )


def _axis(lower: float, upper: float, step: float) -> np.ndarray:
    """Temperature axis starting at zero and running past the response region."""
    if step <= 0:
        raise ValueError("step must be positive.")
    start = min(0.0, lower - step)
    return np.round(np.arange(start, upper + step, step), 6)


def _build(
    intensity: np.ndarray,
    paa: np.ndarray,
    impf_id: int,
    name: str,
    intensity_unit: str,
) -> ImpactFuncSet:
    """Wrap a tabulated response into a checked single-function set.

    The response sits in ``paa`` with ``mdd`` held at 1, per this module's
    convention.
    """
    impf = ImpactFunc(
        haz_type=HAZ_TYPE,
        id=impf_id,
        name=name,
        intensity_unit=intensity_unit,
        intensity=np.asarray(intensity, dtype=float),
        mdd=np.ones_like(intensity, dtype=float),
        paa=np.asarray(paa, dtype=float),
    )
    impf_set = ImpactFuncSet([impf])
    impf_set.check()
    return impf_set


def impf_set_for_metric(metric: str, **kwargs: Any) -> ImpactFuncSet:
    """Build the vulnerability curve for a named heat metric.

    Parameters
    ----------
    metric : str
        Key into :py:data:`METRICS`.
    **kwargs
        Passed to the metric's curve builder.

    Returns
    -------
    climada.entity.ImpactFuncSet

    Raises
    ------
    KeyError
        If ``metric`` is not a known heat metric.
    """
    builders = {
        "mortality": mortality_impf_set,
        "days_above": days_above_impf_set,
        "degree_days": degree_days_impf_set,
        "labour": labour_loss_impf_set,
    }
    if metric not in builders:
        raise KeyError(
            f"Unknown heat metric '{metric}'. Known: {', '.join(sorted(builders))}."
        )
    return builders[metric](**kwargs)


# --------------------------------------------------------------------------- #
# Hazard: frequency and ingest
# --------------------------------------------------------------------------- #


def annual_frequency(hazard: Hazard) -> float:
    """The per-year frequency each event carries if the set is a time series.

    A record of daily temperature fields sampled over ``n`` years contains
    events that each happened once in ``n`` years, so each carries a frequency
    of ``1/n`` per year. Getting this wrong is the standard way to produce heat
    numbers that are out by a factor of the record length.

    The record length is the number of distinct calendar years the events fall
    in, not the raw span between the first and last day. That is what makes
    warm-season data work: twenty Junes-to-Septembers sample twenty years, even
    though the first and last day are only about 19.3 years apart. Pass
    ``years`` to :py:func:`set_annual_frequency` when this reading is wrong for
    your data -- for instance a record of a single winter straddling New Year.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Hazard whose ``date`` array spans the record.

    Returns
    -------
    float
        ``1 / years_sampled``, or NaN when the dates cannot be read.
    """
    dates = np.asarray(hazard.date)
    valid = dates[dates > 0] if dates.size else dates
    if valid.size < 1:
        return float("nan")

    years = pd.DatetimeIndex(
        [pd.Timestamp.fromordinal(int(value)) for value in np.unique(valid)]
    ).year
    sampled = float(len(np.unique(years)))
    if sampled <= 0:
        return float("nan")
    return 1.0 / sampled


def set_annual_frequency(
    hazard: Hazard, years: Optional[float] = None, in_place: bool = False
) -> Hazard:
    """Give every event the frequency implied by the length of the record.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Hazard to reweight.
    years : float, optional
        Number of years the record samples. Default: derive it from the event
        dates, counting distinct calendar years.
    in_place : bool, optional
        Modify ``hazard`` rather than a copy. Default: False.

    Returns
    -------
    climada.hazard.Hazard
        The hazard with ``frequency`` set to ``1 / years`` for every event and
        ``frequency_unit`` set to ``'1/year'``.

    Raises
    ------
    ValueError
        If the record length is neither given nor derivable, or is not
        positive.
    """
    if years is None:
        frequency = annual_frequency(hazard)
        if not np.isfinite(frequency):
            raise ValueError(
                "Cannot derive the record length from the event dates. Pass "
                "'years' explicitly."
            )
    else:
        if years <= 0:
            raise ValueError(f"years must be positive; got {years}.")
        frequency = 1.0 / float(years)

    target = hazard if in_place else copy.deepcopy(hazard)
    target.frequency = np.full(hazard.size, frequency, dtype=float)
    target.frequency_unit = "1/year"
    return target


def hazard_from_dataset(
    dataset,
    intensity: str,
    intensity_unit: str = DEFAULT_INTENSITY_UNIT,
    coordinate_vars: Optional[Dict[str, str]] = None,
    to_celsius: bool = False,
    years: Optional[float] = None,
) -> Hazard:
    """Read gridded temperature from an xarray dataset into a heat hazard.

    Intended for daily maximum temperature from a reanalysis or a climate
    model -- the shape of data that comes out of ERA5 or CMIP6.

    Parameters
    ----------
    dataset : xarray.Dataset
        Must carry a time dimension and latitude/longitude coordinates.
    intensity : str
        Name of the temperature variable, e.g. ``'tasmax'`` or ``'t2m'``.
    intensity_unit : str, optional
        Unit of the temperature after any conversion. Default: ``'degC'``.
    coordinate_vars : dict, optional
        Mapping from CLIMADA's coordinate names (``event``, ``latitude``,
        ``longitude``) to the dataset's, when they differ from the defaults
        (``time``, ``latitude``, ``longitude``).
    to_celsius : bool, optional
        Subtract 273.15 from the data first, for sources in kelvin. Default:
        False.
    years : float, optional
        Length of the record in years, for the event frequency. Default:
        derive it from the dates.

    Returns
    -------
    climada.hazard.Hazard
        Of type ``HW``, with per-year event frequencies.

    Notes
    -----
    Every time step becomes one event. A 30-year daily record is about 11,000
    events, which is large but tractable; subset to the warm season first if
    memory is tight, and pass ``years`` so the frequency still reflects the
    full record.
    """
    data = dataset
    if to_celsius:
        data = dataset.copy()
        data[intensity] = data[intensity] - 273.15

    hazard = Hazard.from_xarray_raster(
        data,
        hazard_type=HAZ_TYPE,
        intensity_unit=intensity_unit,
        intensity=intensity,
        coordinate_vars=coordinate_vars,
    )
    hazard = set_annual_frequency(hazard, years=years, in_place=True)
    hazard.check()
    return hazard


def hazard_temperature_summary(hazard: Hazard) -> Dict[str, float]:
    """Descriptive statistics of a heat hazard's intensity field.

    Parameters
    ----------
    hazard : climada.hazard.Hazard

    Returns
    -------
    dict
        ``peak``, ``mean_of_event_max``, ``events``, ``centroids``,
        ``years`` and ``events_per_year``. Values are NaN where the event
        dates do not permit a calculation.
    """
    intensity = hazard.intensity
    peak = float(intensity.max()) if intensity.nnz else float("nan")

    event_max = np.asarray(intensity.max(axis=1).todense()).ravel()
    frequency = annual_frequency(hazard)
    years = (
        1.0 / frequency if np.isfinite(frequency) and frequency > 0 else float("nan")
    )

    return {
        "peak": peak,
        "mean_of_event_max": (
            float(np.mean(event_max)) if event_max.size else float("nan")
        ),
        "events": float(hazard.size),
        "centroids": float(hazard.centroids.size),
        "years": years,
        "events_per_year": (
            float(hazard.size) / years
            if np.isfinite(years) and years > 0
            else float("nan")
        ),
    }


def days_above_threshold(hazard: Hazard, threshold: float) -> pd.DataFrame:
    """How often each location reaches a temperature, per year.

    A hazard-side diagnostic that needs no exposure or vulnerability, useful
    for sanity-checking a temperature dataset before running an impact.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
    threshold : float
        Temperature at and above which a day counts.

    Returns
    -------
    pandas.DataFrame
        Columns ``latitude``, ``longitude`` and ``days_per_year``.
    """
    # Count on the stored entries only. Comparing a sparse matrix against a
    # scalar densifies it, which on a real reanalysis grid (thousands of days
    # by tens of thousands of points) is gigabytes for a column count.
    matrix = hazard.intensity.tocoo()
    n_centroids = hazard.centroids.size
    hot = matrix.data >= threshold
    counts = np.bincount(matrix.col[hot], minlength=n_centroids).astype(float)

    if threshold <= 0:
        # Cells with no stored value are zero, which clears a threshold at or
        # below zero. Counting only stored entries would miss them.
        stored = np.bincount(matrix.col, minlength=n_centroids)
        counts += hazard.size - stored

    frequency = annual_frequency(hazard)
    years = 1.0 / frequency if np.isfinite(frequency) and frequency > 0 else np.nan

    return pd.DataFrame(
        {
            "latitude": hazard.centroids.lat,
            "longitude": hazard.centroids.lon,
            "days_per_year": counts / years if years and np.isfinite(years) else counts,
        }
    )


# --------------------------------------------------------------------------- #
# Adaptation measures
# --------------------------------------------------------------------------- #

HEAT_MEASURE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Urban greening": {
        "note": (
            "Tree canopy and green space cool the air locally. Entered as a "
            "negative intensity offset: the assets behave as though the day "
            "were cooler."
        ),
        "fields": {"hazard_inten_imp_b": -1.5},
    },
    "Cool roofs and surfaces": {
        "note": (
            "High-albedo roofs and pavements cut the urban heat island. Also "
            "a negative intensity offset, typically smaller than greening."
        ),
        "fields": {"hazard_inten_imp_b": -1.0},
    },
    "Heat-health warning system": {
        "note": (
            "Alerts plus a public-health response cut the harm rate at a "
            "given temperature; the literature supports roughly a third."
        ),
        "fields": {"paa_impact_a": 0.7},
    },
    "Cooling centres": {
        "note": (
            "Air-conditioned refuges for the most exposed. Smaller population "
            "reach than a warning system, so a smaller reduction."
        ),
        "fields": {"paa_impact_a": 0.85},
    },
    "Home cooling retrofit": {
        "note": (
            "Insulation, shutters and ventilation in dwellings. Cuts the harm "
            "rate substantially but costs far more per household."
        ),
        "fields": {"paa_impact_a": 0.55},
    },
}
"""Heat adaptation options, as measure rows the interface can drop into its table.

Reductions are indicative and vary widely by city, housing stock and
population age structure. They are a starting point to be replaced with local
evidence, not defaults to report.
"""


# --------------------------------------------------------------------------- #
# Synthetic demonstration data
# --------------------------------------------------------------------------- #


def demo_heat_hazard(
    years: int = 20,
    centre: Tuple[float, float] = (41.39, 2.17),
    grid: int = 12,
    extent: float = 0.22,
    summer_mean: float = 28.0,
    warming_per_decade: float = 0.4,
    seed: int = 20240501,
) -> Hazard:
    """A synthetic daily-temperature event set, for demonstrating the workflow.

    **This is not a climatology.** It is a reproducible statistical toy: a
    warm-season daily maximum temperature built from a seasonal cycle, an
    urban heat island that peaks at the centre of the grid, day-to-day
    persistence, and a warming trend. Use it to learn the interface and to
    test a configuration end to end; replace it with reanalysis or model
    output before drawing any conclusion about a real place.

    Parameters
    ----------
    years : int, optional
        Number of warm seasons to generate. Default: 20.
    centre : tuple of float, optional
        ``(latitude, longitude)`` of the grid centre. Default: Barcelona.
    grid : int, optional
        Points per side of the square grid. Default: 12, i.e. 144 centroids.
    extent : float, optional
        Half-width of the grid in degrees. Default: 0.22, about 25 km across.
    summer_mean : float, optional
        Mean warm-season daily maximum at the grid edge, in degC. Default: 28.
    warming_per_decade : float, optional
        Linear trend added over the record, in degC per decade. Default: 0.4.
    seed : int, optional
        Random seed, so the demo is identical on every run.

    Returns
    -------
    climada.hazard.Hazard
        Of type ``HW``, one event per day of each warm season, with per-year
        frequencies already set.
    """
    rng = np.random.default_rng(seed)
    season_days = 122  # June to September inclusive.

    lats = np.linspace(centre[0] - extent, centre[0] + extent, grid)
    lons = np.linspace(centre[1] - extent, centre[1] + extent, grid)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    lat_flat, lon_flat = lat_mesh.ravel(), lon_mesh.ravel()

    # Urban heat island: a Gaussian bump of up to 3 degC at the grid centre.
    radius = np.sqrt(
        ((lat_flat - centre[0]) / extent) ** 2 + ((lon_flat - centre[1]) / extent) ** 2
    )
    heat_island = 3.0 * np.exp(-((radius / 0.6) ** 2))

    n_events = years * season_days
    intensity = np.empty((n_events, lat_flat.size), dtype=np.float32)
    dates = np.empty(n_events, dtype=int)

    anomaly = 0.0
    index = 0
    for year in range(years):
        trend = warming_per_decade * year / 10.0
        for day in range(season_days):
            # Seasonal cycle peaking in the first half of August.
            seasonal = 4.5 * np.sin(np.pi * day / season_days)
            # Day-to-day persistence: an AR(1) weather anomaly.
            anomaly = 0.82 * anomaly + rng.normal(0.0, 1.7)
            field = (
                summer_mean
                + trend
                + seasonal
                + anomaly
                + heat_island
                + rng.normal(0.0, 0.3, lat_flat.size)
            )
            intensity[index] = field
            dates[index] = (
                pd.Timestamp(year=2004 + year, month=6, day=1).toordinal() + day
            )
            index += 1

    hazard = Hazard(
        haz_type=HAZ_TYPE,
        units=DEFAULT_INTENSITY_UNIT,
        intensity=_sparse(intensity),
        centroids=_centroids(lat_flat, lon_flat),
        event_id=np.arange(1, n_events + 1),
        event_name=[
            pd.Timestamp.fromordinal(int(d)).strftime("%Y-%m-%d") for d in dates
        ],
        date=dates,
        frequency=np.full(n_events, 1.0 / years),
        frequency_unit="1/year",
        orig=np.ones(n_events, dtype=bool),
    )
    hazard.check()
    return hazard


def demo_population(
    hazard: Hazard,
    total_population: float = 1_600_000.0,
    ref_year: int = 2024,
) -> Exposures:
    """A synthetic population grid matching a demo heat hazard's centroids.

    People are concentrated towards the centre of the grid, on the same
    Gaussian as the demo hazard's urban heat island -- which is the
    uncomfortable point about heat risk in cities: the hottest places hold the
    most people.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Supplies the grid. Usually the output of :py:func:`demo_heat_hazard`.
    total_population : float, optional
        People to distribute over the grid. Default: 1.6 million.
    ref_year : int, optional
        Reference year of the population figures. Default: 2024.

    Returns
    -------
    climada.entity.Exposures
        Values in people, with an ``impf_HW`` column already set to 1.
    """
    lat = np.asarray(hazard.centroids.lat, dtype=float)
    lon = np.asarray(hazard.centroids.lon, dtype=float)
    centre_lat, centre_lon = lat.mean(), lon.mean()
    span = max(np.ptp(lat), np.ptp(lon)) / 2.0 or 1.0

    radius = np.sqrt(
        ((lat - centre_lat) / span) ** 2 + ((lon - centre_lon) / span) ** 2
    )
    density = np.exp(-((radius / 0.5) ** 2)) + 0.05
    people = total_population * density / density.sum()

    exposures = Exposures(
        pd.DataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "value": people,
                # 1 is the dense, hot centre; 2 the cooler periphery.
                "region_id": np.where(radius < 0.5, 1, 2),
            }
        ),
        value_unit="people",
        ref_year=ref_year,
    )
    exposures.gdf[f"impf_{HAZ_TYPE}"] = 1
    exposures.check()
    return exposures


def _sparse(array: np.ndarray):
    """Dense temperature field as the sparse matrix Hazard expects.

    Temperatures are dense by nature, so nothing is saved by sparsity here;
    the conversion exists because ``Hazard.intensity`` is a sparse matrix.
    """
    from scipy import sparse

    return sparse.csr_matrix(array)


def _centroids(lat: np.ndarray, lon: np.ndarray):
    """Centroids for a flat lat/lon grid."""
    from climada.hazard.centroids import Centroids

    return Centroids(lat=lat, lon=lon)


DEMO_MMT = 27.0
"""Minimum-mortality temperature used by the demo, matched to its warm climate.

The demo grid is a Mediterranean-like city whose warm-season days average
around 32 degC. An MMT drawn from a cooler climate would attribute deaths to
ordinary summer weather, so the demo uses a locally plausible threshold.
"""


def demo_bundle(years: int = 20, metric: str = "mortality") -> Dict[str, Any]:
    """A complete synthetic heat analysis: hazard, population and a curve.

    Parameters
    ----------
    years : int, optional
        Warm seasons to generate. Default: 20.
    metric : str, optional
        Which heat metric to build the vulnerability curve for. Default:
        ``'mortality'``.

    Returns
    -------
    dict
        ``hazard``, ``exposures``, ``impf_set``, ``metric`` and ``note``.
    """
    hazard = demo_heat_hazard(years=years)
    exposures = demo_population(hazard)
    if metric == "mortality":
        impf_set = mortality_impf_set(mmt=DEMO_MMT)
    else:
        impf_set = impf_set_for_metric(metric)
    return {
        "hazard": hazard,
        "exposures": exposures,
        "impf_set": impf_set,
        "metric": metric,
        "note": (
            "Synthetic demonstration data, not a climatology. The temperature "
            "field is a seeded statistical toy with a seasonal cycle, an urban "
            "heat island and a warming trend; the population is a smooth "
            "gradient towards the centre. Use it to learn the workflow, then "
            "replace it with reanalysis or model output."
        ),
    }
