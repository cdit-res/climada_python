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

Analysis layer of the CLIMADA user interface.

This module holds every computation the interface performs, expressed as plain
functions over CLIMADA objects. It imports no Streamlit, so the same calls can
be made from a script or notebook -- the interface is a thin shell around it.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from climada.engine import ImpactCalc
from climada.engine.cost_benefit import (
    CostBenefit,
    risk_aai_agg,
    risk_rp_100,
    risk_rp_250,
)
from climada.engine.impact import Impact
from climada.entity import (
    DiscRates,
    Entity,
    Exposures,
    ImpactFuncSet,
    Measure,
    MeasureSet,
)
from climada.hazard import Hazard
from climada.ui.formatting import safe_ratio
from climada.util import dates_times as u_dt

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RETURN_PERIODS",
    "RISK_FUNCTIONS",
    "RiskResult",
    "WaterfallResult",
    "annual_impact_table",
    "build_disc_rates",
    "build_entity",
    "build_measure",
    "build_measure_set",
    "compute_risk",
    "cost_benefit_table",
    "default_measure_row",
    "exceedance_table",
    "grow_exposures",
    "impact_by_region",
    "impact_point_table",
    "measure_impact_curves",
    "run_cost_benefit",
    "scale_hazard",
    "top_events_table",
    "total_exposure_value",
    "waterfall_components",
]


DEFAULT_RETURN_PERIODS: Tuple[int, ...] = (5, 10, 25, 50, 100, 250, 500)
"""Return periods reported by default in the risk view."""

RISK_FUNCTIONS: Dict[str, Callable[[Impact], float]] = {
    "Average annual impact": risk_aai_agg,
    "100-year return period impact": risk_rp_100,
    "250-year return period impact": risk_rp_250,
}
"""Risk metrics selectable as the basis of the cost-benefit calculation."""

NO_MEASURE = "no measure"
"""Key CLIMADA uses for the unmitigated case in :py:class:`CostBenefit` results."""


# --------------------------------------------------------------------------- #
# Physical risk
# --------------------------------------------------------------------------- #


@dataclass
class RiskResult:
    """Outcome of a physical risk assessment.

    Attributes
    ----------
    impact : climada.engine.impact.Impact
        The underlying CLIMADA impact object.
    haz_type : str
        Hazard type the impact was computed for.
    unit : str
        Value unit of the impact, taken from the exposures.
    total_value : float
        Sum of the exposure values.
    aai : float
        Average annual impact, aggregated over all exposure points.
    return_periods : numpy.ndarray
        Return periods at which ``rp_impact`` is reported.
    rp_impact : numpy.ndarray
        Impact exceeded once every corresponding return period.
    freq_curve_return_per : numpy.ndarray
        Return periods of the full exceedance frequency curve.
    freq_curve_impact : numpy.ndarray
        Impacts of the full exceedance frequency curve.
    """

    impact: Impact
    haz_type: str
    unit: str
    total_value: float
    aai: float
    return_periods: np.ndarray
    rp_impact: np.ndarray
    freq_curve_return_per: np.ndarray
    freq_curve_impact: np.ndarray

    @property
    def loss_ratio(self) -> float:
        """Average annual impact as a fraction of the total exposure value."""
        return safe_ratio(self.aai, self.total_value)

    @property
    def max_event_impact(self) -> float:
        """Largest single-event impact in the hazard event set."""
        if self.impact.at_event.size == 0:
            return float("nan")
        return float(np.max(self.impact.at_event))

    def rp_value(self, return_period: float) -> float:
        """Impact at one return period, interpolated on the exceedance curve.

        Parameters
        ----------
        return_period : float
            Return period in years.

        Returns
        -------
        float
            Impact exceeded once every ``return_period`` years, or NaN if the
            event set is empty.
        """
        if self.freq_curve_return_per.size == 0:
            return float("nan")
        return float(
            np.interp(return_period, self.freq_curve_return_per, self.freq_curve_impact)
        )


def total_exposure_value(exposures: Exposures) -> float:
    """Total value carried by an exposures object.

    Parameters
    ----------
    exposures : climada.entity.Exposures

    Returns
    -------
    float
        Sum of the ``value`` column, or NaN when the column is absent.
    """
    values = exposures.value
    if values is None:
        return float("nan")
    return float(np.nansum(values))


def compute_risk(
    exposures: Exposures,
    impf_set: ImpactFuncSet,
    hazard: Hazard,
    return_periods: Sequence[float] = DEFAULT_RETURN_PERIODS,
    save_mat: bool = False,
    assign_centroids: bool = True,
) -> RiskResult:
    """Run a physical risk assessment: exposures x impact functions x hazard.

    Parameters
    ----------
    exposures : climada.entity.Exposures
        Assets at risk. Must carry an impact function column for the hazard type.
    impf_set : climada.entity.ImpactFuncSet
        Vulnerability curves.
    hazard : climada.hazard.Hazard
        Probabilistic or historic event set.
    return_periods : sequence of float, optional
        Return periods to report. Default: :py:data:`DEFAULT_RETURN_PERIODS`.
    save_mat : bool, optional
        Keep the full event-by-asset impact matrix. Memory-hungry on large
        exposures; only needed for per-event asset detail. Default: False.
    assign_centroids : bool, optional
        Assign hazard centroids to the exposures first. Skip when they are
        already assigned for this hazard. Default: True.

    Returns
    -------
    RiskResult
    """
    impact = ImpactCalc(exposures, impf_set, hazard).impact(
        save_mat=save_mat, assign_centroids=assign_centroids
    )
    curve = impact.calc_freq_curve()
    rp_array = np.asarray(return_periods, dtype=float)
    rp_impact = (
        np.interp(rp_array, curve.return_per, curve.impact)
        if curve.return_per.size
        else np.full(rp_array.shape, np.nan)
    )
    return RiskResult(
        impact=impact,
        haz_type=hazard.haz_type,
        unit=impact.unit or exposures.value_unit,
        total_value=total_exposure_value(exposures),
        aai=float(impact.aai_agg),
        return_periods=rp_array,
        rp_impact=rp_impact,
        freq_curve_return_per=np.asarray(curve.return_per, dtype=float),
        freq_curve_impact=np.asarray(curve.impact, dtype=float),
    )


def exceedance_table(risk: RiskResult) -> pd.DataFrame:
    """Return-period losses as a table.

    Parameters
    ----------
    risk : RiskResult

    Returns
    -------
    pandas.DataFrame
        Columns ``Return period (years)``, ``Exceedance probability``,
        ``Impact`` and ``Impact / total value``.
    """
    return pd.DataFrame(
        {
            "Return period (years)": risk.return_periods,
            "Exceedance probability": 1.0 / risk.return_periods,
            f"Impact ({risk.unit})": risk.rp_impact,
            "Impact / total value": [
                safe_ratio(value, risk.total_value) for value in risk.rp_impact
            ],
        }
    )


def top_events_table(impact: Impact, limit: int = 10) -> pd.DataFrame:
    """The most damaging events in the set, largest first.

    Parameters
    ----------
    impact : climada.engine.impact.Impact
    limit : int, optional
        Maximum number of rows. Default: 10.

    Returns
    -------
    pandas.DataFrame
        Columns ``Event``, ``Date``, ``Frequency (1/year)``, ``Return period
        (years)`` and ``Impact``. Empty when the impact has no events.
    """
    at_event = np.asarray(impact.at_event, dtype=float)
    if at_event.size == 0:
        return pd.DataFrame(
            columns=[
                "Event",
                "Date",
                "Frequency (1/year)",
                "Return period (years)",
                f"Impact ({impact.unit})",
            ]
        )

    order = np.argsort(at_event)[::-1][:limit]
    names = list(impact.event_name) if len(impact.event_name) == at_event.size else []
    dates = np.asarray(impact.date) if np.size(impact.date) == at_event.size else None
    freq = np.asarray(impact.frequency, dtype=float)

    rows = []
    for idx in order:
        frequency = float(freq[idx]) if freq.size == at_event.size else float("nan")
        rows.append(
            {
                "Event": names[idx] if names else str(impact.event_id[idx]),
                "Date": _format_date(dates[idx]) if dates is not None else "",
                "Frequency (1/year)": frequency,
                "Return period (years)": safe_ratio(1.0, frequency),
                f"Impact ({impact.unit})": at_event[idx],
            }
        )
    return pd.DataFrame(rows)


def _format_date(ordinal: Any) -> str:
    """Render a CLIMADA ordinal date, tolerating placeholder values."""
    try:
        if int(ordinal) <= 0:
            return ""
        return u_dt.date_to_str(int(ordinal))
    except (ValueError, TypeError, OverflowError):
        return ""


def impact_point_table(impact: Impact, limit: Optional[int] = None) -> pd.DataFrame:
    """Expected annual impact per exposure point, for mapping.

    Parameters
    ----------
    impact : climada.engine.impact.Impact
    limit : int, optional
        Keep only the ``limit`` highest-impact points, so that very large
        exposure sets stay renderable. Default: keep all.

    Returns
    -------
    pandas.DataFrame
        Columns ``latitude``, ``longitude`` and ``eai``, sorted by ``eai``.
    """
    coords = np.asarray(impact.coord_exp)
    eai = np.asarray(impact.eai_exp, dtype=float)
    if coords.size == 0 or eai.size == 0:
        return pd.DataFrame(columns=["latitude", "longitude", "eai"])

    frame = pd.DataFrame(
        {"latitude": coords[:, 0], "longitude": coords[:, 1], "eai": eai}
    ).sort_values("eai", ascending=False, ignore_index=True)
    if limit is not None and len(frame) > limit:
        frame = frame.iloc[:limit].reset_index(drop=True)
    return frame


def impact_by_region(impact: Impact, exposures: Exposures) -> Optional[pd.DataFrame]:
    """Aggregate the expected annual impact by the exposures' ``region_id``.

    Parameters
    ----------
    impact : climada.engine.impact.Impact
    exposures : climada.entity.Exposures

    Returns
    -------
    pandas.DataFrame or None
        Columns ``Region``, ``Exposed value``, ``Expected annual impact`` and
        ``Loss ratio``; ``None`` when the exposures carry no ``region_id`` or
        its length does not match the impact.
    """
    gdf = exposures.gdf
    if "region_id" not in gdf.columns:
        return None
    eai = np.asarray(impact.eai_exp, dtype=float)
    values = np.asarray(gdf["value"].values, dtype=float)
    if eai.size != len(gdf):
        return None

    frame = pd.DataFrame(
        {"Region": gdf["region_id"].values, "Exposed value": values, "eai": eai}
    )
    grouped = frame.groupby("Region", as_index=False).sum(numeric_only=True)
    grouped = grouped.rename(columns={"eai": "Expected annual impact"})
    grouped["Loss ratio"] = [
        safe_ratio(row["Expected annual impact"], row["Exposed value"])
        for _, row in grouped.iterrows()
    ]
    return grouped.sort_values(
        "Expected annual impact", ascending=False, ignore_index=True
    )


def annual_impact_table(impact: Impact) -> Optional[pd.DataFrame]:
    """Impact summed per calendar year, when the events carry usable dates.

    Parameters
    ----------
    impact : climada.engine.impact.Impact

    Returns
    -------
    pandas.DataFrame or None
        Columns ``Year`` and ``Impact``; ``None`` when the event set has no
        dates to group on.
    """
    try:
        year_set = impact.impact_per_year(all_years=True)
    except (AttributeError, ValueError, TypeError, OverflowError):
        return None
    if not year_set:
        return None
    frame = pd.DataFrame(
        {
            "Year": list(year_set.keys()),
            f"Impact ({impact.unit})": list(year_set.values()),
        }
    )
    return frame.sort_values("Year", ignore_index=True)


# --------------------------------------------------------------------------- #
# Adaptation measures
# --------------------------------------------------------------------------- #

MEASURE_COLUMNS: Tuple[str, ...] = (
    "name",
    "cost",
    "hazard_inten_imp_a",
    "hazard_inten_imp_b",
    "mdd_impact_a",
    "mdd_impact_b",
    "paa_impact_a",
    "paa_impact_b",
    "hazard_freq_cutoff",
    "risk_transf_attach",
    "risk_transf_cover",
    "risk_transf_cost_factor",
)
"""Editable fields of a measure, in the order the interface presents them."""


def default_measure_row(name: str = "New measure") -> Dict[str, Any]:
    """A measure specification with no effect, ready to be edited.

    Parameters
    ----------
    name : str, optional
        Measure name. Default: ``'New measure'``.

    Returns
    -------
    dict
        Keys as in :py:data:`MEASURE_COLUMNS`.
    """
    return {
        "name": name,
        "cost": 0.0,
        "hazard_inten_imp_a": 1.0,
        "hazard_inten_imp_b": 0.0,
        "mdd_impact_a": 1.0,
        "mdd_impact_b": 0.0,
        "paa_impact_a": 1.0,
        "paa_impact_b": 0.0,
        "hazard_freq_cutoff": 0.0,
        "risk_transf_attach": 0.0,
        "risk_transf_cover": 0.0,
        "risk_transf_cost_factor": 1.0,
    }


def build_measure(
    row: Dict[str, Any], haz_type: str, color: Optional[Sequence[float]] = None
) -> Measure:
    """Turn an edited measure row into a CLIMADA :py:class:`Measure`.

    The ``a``/``b`` pairs follow CLIMADA's convention: the impact function's
    intensity axis becomes ``a * intensity - b`` and its ``mdd``/``paa`` become
    ``a * value + b``. Note the sign on the intensity offset -- shifting the
    curve to the right is what protects the assets, so a measure that keeps
    4 m/s of wind off them is ``hazard_inten_imp = (1, -4)``, and a positive
    offset increases damage.

    Parameters
    ----------
    row : dict
        Measure specification, keys as in :py:data:`MEASURE_COLUMNS`.
    haz_type : str
        Hazard type the measure applies to, e.g. ``'TC'``.
    color : sequence of float, optional
        RGB triple in [0, 1] used in CLIMADA's own plots. Default: mid grey.

    Returns
    -------
    climada.entity.Measure
    """
    return Measure(
        name=str(row["name"]),
        haz_type=haz_type,
        cost=float(row.get("cost", 0.0) or 0.0),
        hazard_inten_imp=(
            float(row.get("hazard_inten_imp_a", 1.0) or 0.0),
            float(row.get("hazard_inten_imp_b", 0.0) or 0.0),
        ),
        mdd_impact=(
            float(row.get("mdd_impact_a", 1.0) or 0.0),
            float(row.get("mdd_impact_b", 0.0) or 0.0),
        ),
        paa_impact=(
            float(row.get("paa_impact_a", 1.0) or 0.0),
            float(row.get("paa_impact_b", 0.0) or 0.0),
        ),
        hazard_freq_cutoff=float(row.get("hazard_freq_cutoff", 0.0) or 0.0),
        risk_transf_attach=float(row.get("risk_transf_attach", 0.0) or 0.0),
        risk_transf_cover=float(row.get("risk_transf_cover", 0.0) or 0.0),
        risk_transf_cost_factor=float(row.get("risk_transf_cost_factor", 1.0) or 1.0),
        color_rgb=np.array(
            color if color is not None else (0.4, 0.4, 0.4), dtype=float
        ),
    )


def build_measure_set(
    rows: Sequence[Dict[str, Any]],
    haz_type: str,
    colors: Optional[Sequence[Sequence[float]]] = None,
) -> MeasureSet:
    """Assemble a :py:class:`MeasureSet` from edited measure rows.

    Parameters
    ----------
    rows : sequence of dict
        Measure specifications.
    haz_type : str
        Hazard type the measures apply to.
    colors : sequence of RGB triples, optional
        One colour per row. Default: mid grey for all.

    Returns
    -------
    climada.entity.MeasureSet

    Raises
    ------
    ValueError
        If two measures share a name, which CLIMADA cannot disambiguate.
    """
    names = [str(row["name"]).strip() for row in rows]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Measure names must be unique; repeated: {', '.join(sorted(duplicates))}"
        )
    if any(not name for name in names):
        raise ValueError("Every measure needs a name.")

    measure_set = MeasureSet()
    for index, row in enumerate(rows):
        color = colors[index] if colors is not None and index < len(colors) else None
        measure_set.append(build_measure(row, haz_type, color))
    measure_set.check()
    return measure_set


def measure_impact_curves(
    measure: Measure,
    exposures: Exposures,
    impf_set: ImpactFuncSet,
    hazard: Hazard,
) -> Tuple[Exposures, ImpactFuncSet, Hazard]:
    """Apply a measure and return the modified inputs, without computing impact.

    Useful for showing the user what a measure actually does to the
    vulnerability curve before running the full calculation.

    Parameters
    ----------
    measure : climada.entity.Measure
    exposures : climada.entity.Exposures
    impf_set : climada.entity.ImpactFuncSet
    hazard : climada.hazard.Hazard

    Returns
    -------
    tuple
        ``(new_exposures, new_impact_functions, new_hazard)``.
    """
    return measure.apply(exposures, impf_set, hazard)


# --------------------------------------------------------------------------- #
# Discount rates, scenarios and entities
# --------------------------------------------------------------------------- #


def build_disc_rates(rate: float, year_start: int, year_end: int) -> DiscRates:
    """Constant discount rate over a horizon.

    Parameters
    ----------
    rate : float
        Annual discount rate as a fraction, e.g. 0.02 for 2%.
    year_start : int
        First year of the horizon, inclusive.
    year_end : int
        Last year of the horizon, inclusive.

    Returns
    -------
    climada.entity.DiscRates
    """
    years = np.arange(int(year_start), int(year_end) + 1)
    return DiscRates(years=years, rates=np.full(years.shape, float(rate)))


def disc_rates_from_table(frame: pd.DataFrame) -> DiscRates:
    """Discount rates from an edited year/rate table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Must have columns ``year`` and ``rate``.

    Returns
    -------
    climada.entity.DiscRates
    """
    clean = frame.dropna(subset=["year", "rate"]).sort_values("year")
    disc = DiscRates(
        years=clean["year"].astype(int).values,
        rates=clean["rate"].astype(float).values,
    )
    disc.check()
    return disc


def scale_hazard(
    hazard: Hazard,
    intensity_factor: float = 1.0,
    frequency_factor: float = 1.0,
) -> Hazard:
    """A copy of a hazard with intensities and/or frequencies rescaled.

    This is the quick way to express a climate scenario when no downscaled
    future event set is at hand: it stretches every event's intensity and/or
    makes every event more frequent. For a defensible study, prefer a future
    hazard set from the CLIMADA Data API.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Hazard to copy. Left untouched.
    intensity_factor : float, optional
        Multiplier on the intensity matrix. Default: 1.0.
    frequency_factor : float, optional
        Multiplier on the event frequencies. Default: 1.0.

    Returns
    -------
    climada.hazard.Hazard
    """
    scaled = copy.deepcopy(hazard)
    if intensity_factor != 1.0:
        scaled.intensity = scaled.intensity.multiply(float(intensity_factor)).tocsr()
    if frequency_factor != 1.0:
        scaled.frequency = np.asarray(scaled.frequency, dtype=float) * float(
            frequency_factor
        )
    return scaled


def grow_exposures(
    exposures: Exposures,
    growth_factor: float = 1.0,
    ref_year: Optional[int] = None,
) -> Exposures:
    """A copy of the exposures with all values scaled and the year updated.

    Parameters
    ----------
    exposures : climada.entity.Exposures
        Exposures to copy. Left untouched.
    growth_factor : float, optional
        Multiplier on every asset value, e.g. 1.4 for 40% growth. Default: 1.0.
    ref_year : int, optional
        Reference year of the copy. Default: keep the original.

    Returns
    -------
    climada.entity.Exposures
    """
    grown = exposures.copy(deep=True)
    if growth_factor != 1.0:
        grown.gdf["value"] = grown.gdf["value"].astype(float) * float(growth_factor)
    if ref_year is not None:
        grown.ref_year = int(ref_year)
    return grown


def build_entity(
    exposures: Exposures,
    impf_set: ImpactFuncSet,
    measure_set: MeasureSet,
    disc_rates: DiscRates,
    ref_year: Optional[int] = None,
) -> Entity:
    """Assemble an :py:class:`Entity` and validate it.

    Parameters
    ----------
    exposures : climada.entity.Exposures
    impf_set : climada.entity.ImpactFuncSet
    measure_set : climada.entity.MeasureSet
    disc_rates : climada.entity.DiscRates
    ref_year : int, optional
        Override the exposures' reference year. Default: keep it.

    Returns
    -------
    climada.entity.Entity
    """
    exp = exposures.copy(deep=True)
    if ref_year is not None:
        exp.ref_year = int(ref_year)
    entity = Entity(
        exposures=exp,
        disc_rates=disc_rates,
        impact_func_set=impf_set,
        measure_set=measure_set,
    )
    entity.check()
    return entity


# --------------------------------------------------------------------------- #
# Cost-benefit analysis
# --------------------------------------------------------------------------- #


def run_cost_benefit(
    hazard: Hazard,
    entity: Entity,
    haz_future: Optional[Hazard] = None,
    ent_future: Optional[Entity] = None,
    future_year: Optional[int] = None,
    risk_func: Callable[[Impact], float] = risk_aai_agg,
    imp_time_depen: Optional[float] = None,
    save_imp: bool = True,
) -> CostBenefit:
    """Run CLIMADA's cost-benefit calculation with the interface's guardrails.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Present-day hazard.
    entity : climada.entity.Entity
        Present-day entity, carrying the measures and discount rates.
    haz_future : climada.hazard.Hazard, optional
        Future hazard. Default: none, i.e. no change in hazard.
    ent_future : climada.entity.Entity, optional
        Future entity. Its exposures' reference year sets the horizon.
    future_year : int, optional
        End of the appraisal horizon when no future entity is given.
    risk_func : callable, optional
        Risk metric benefits are measured in. Default: average annual impact.
    imp_time_depen : float, optional
        Exponent of the impact growth path between present and future. 1 is
        linear, below 1 front-loads the change, above 1 back-loads it.
    save_imp : bool, optional
        Keep per-measure impacts, needed to combine measures or layer risk
        transfer afterwards. Default: True.

    Returns
    -------
    climada.engine.cost_benefit.CostBenefit

    Raises
    ------
    ValueError
        If the horizon does not extend beyond the present year, which would
        make every benefit undefined.
    """
    present_year = int(entity.exposures.ref_year)
    horizon_end = future_year
    if ent_future is not None:
        horizon_end = int(ent_future.exposures.ref_year)
    if horizon_end is None:
        horizon_end = present_year

    if (
        haz_future is not None or ent_future is not None
    ) and horizon_end <= present_year:
        raise ValueError(
            f"The appraisal horizon must end after the present year {present_year}; "
            f"got {horizon_end}. Set a later future year."
        )

    cost_ben = CostBenefit()
    cost_ben.calc(
        hazard,
        entity,
        haz_future=haz_future,
        ent_future=ent_future,
        future_year=future_year,
        risk_func=risk_func,
        imp_time_depen=imp_time_depen,
        save_imp=save_imp,
    )
    return cost_ben


def _benefit_cost_ratio(cost_ben_ratio: float) -> float:
    """Invert CLIMADA's cost/benefit ratio into a benefit/cost one.

    A measure that averts nothing has an infinite cost/benefit ratio and a
    benefit/cost ratio of zero, which is the number a reader expects to see.

    Parameters
    ----------
    cost_ben_ratio : float
        CLIMADA's cost divided by benefit.

    Returns
    -------
    float
        Benefit divided by cost: 0 for a measure with no benefit, infinity for
        a costless one, NaN when the ratio is undefined.
    """
    if cost_ben_ratio is None or np.isnan(cost_ben_ratio):
        return float("nan")
    if np.isinf(cost_ben_ratio):
        return 0.0
    if cost_ben_ratio == 0:
        return float("inf")
    return 1.0 / cost_ben_ratio


def _measure_cost(cost_ben: CostBenefit, name: str) -> float:
    """Cost of a measure, including any risk-transfer premium.

    Recovered from the cost/benefit ratio where that is finite, the way
    CLIMADA's own results table does it, so that the premium charged on a risk
    transfer layer is not silently dropped.

    Parameters
    ----------
    cost_ben : climada.engine.cost_benefit.CostBenefit
    name : str
        Measure name.

    Returns
    -------
    float
    """
    ratio = cost_ben.cost_ben_ratio.get(name, float("nan"))
    benefit = cost_ben.benefit.get(name, float("nan"))
    if np.isfinite(ratio) and ratio != 0:
        return float(ratio * benefit)
    return float(cost_ben.imp_meas_future[name]["cost"][0])


def cost_benefit_table(cost_ben: CostBenefit) -> pd.DataFrame:
    """Per-measure costs, benefits and ratios as a table.

    Cost is recovered the way CLIMADA reports it: from the cost-benefit ratio
    where that is finite, so that risk-transfer premiums are included, and from
    the measure's own cost otherwise.

    Parameters
    ----------
    cost_ben : climada.engine.cost_benefit.CostBenefit
        A calculated cost-benefit object.

    Returns
    -------
    pandas.DataFrame
        One row per measure, sorted by benefit/cost ratio, best first.
    """
    unit = cost_ben.unit
    rows = []
    for name, benefit in cost_ben.benefit.items():
        ratio = cost_ben.cost_ben_ratio.get(name, float("nan"))
        cost = _measure_cost(cost_ben, name)
        rows.append(
            {
                "Measure": name,
                f"Cost ({unit})": cost,
                f"Benefit ({unit})": benefit,
                "Benefit/cost ratio": _benefit_cost_ratio(ratio),
                f"Net benefit ({unit})": benefit - cost,
                "Residual risk share": safe_ratio(
                    cost_ben.imp_meas_future[name]["risk"],
                    cost_ben.imp_meas_future[NO_MEASURE]["risk"],
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        "Benefit/cost ratio", ascending=False, ignore_index=True, na_position="last"
    )


def cost_benefit_summary(cost_ben: CostBenefit) -> Dict[str, float]:
    """Headline figures of a cost-benefit calculation.

    Parameters
    ----------
    cost_ben : climada.engine.cost_benefit.CostBenefit

    Returns
    -------
    dict
        ``total_climate_risk``, ``averted``, ``residual``, ``total_cost``,
        ``portfolio_bcr``, ``annual_risk_future`` and ``annual_risk_present``
        (the last is NaN when only one time point was computed).
    """
    benefits = np.array(list(cost_ben.benefit.values()), dtype=float)
    total_benefit = float(np.nansum(benefits)) if benefits.size else 0.0

    total_cost = 0.0
    for name in cost_ben.benefit:
        cost = _measure_cost(cost_ben, name)
        if np.isfinite(cost):
            total_cost += cost

    present_risk = float("nan")
    if cost_ben.imp_meas_present:
        present_risk = float(cost_ben.imp_meas_present[NO_MEASURE]["risk"])

    return {
        "total_climate_risk": float(cost_ben.tot_climate_risk),
        "averted": total_benefit,
        "residual": float(cost_ben.tot_climate_risk) - total_benefit,
        "total_cost": total_cost,
        "portfolio_bcr": safe_ratio(total_benefit, total_cost),
        "annual_risk_future": float(cost_ben.imp_meas_future[NO_MEASURE]["risk"]),
        "annual_risk_present": present_risk,
        "present_year": int(cost_ben.present_year),
        "future_year": int(cost_ben.future_year),
    }


def measure_risk_table(cost_ben: CostBenefit) -> pd.DataFrame:
    """Residual annual risk under each measure, including the unmitigated case.

    Parameters
    ----------
    cost_ben : climada.engine.cost_benefit.CostBenefit

    Returns
    -------
    pandas.DataFrame
        Columns ``Measure``, the future residual risk, the annual risk averted
        and the share of risk removed.
    """
    baseline = float(cost_ben.imp_meas_future[NO_MEASURE]["risk"])
    unit = cost_ben.unit
    rows = []
    for name, values in cost_ben.imp_meas_future.items():
        residual = float(values["risk"])
        rows.append(
            {
                "Measure": name,
                f"Residual annual risk ({unit})": residual,
                f"Annual risk averted ({unit})": baseline - residual,
                "Share of risk averted": safe_ratio(baseline - residual, baseline),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class WaterfallResult:
    """Decomposition of the change in risk between two points in time.

    Attributes
    ----------
    present_year : int
    future_year : int
    present_risk : float
        Risk today, with today's hazard and today's exposures.
    development_risk : float
        Risk with future exposures but today's hazard.
    future_risk : float
        Risk with future exposures and future hazard.
    unit : str
    """

    present_year: int
    future_year: int
    present_risk: float
    development_risk: float
    future_risk: float
    unit: str

    @property
    def development_delta(self) -> float:
        """Risk added by socio-economic development alone."""
        return self.development_risk - self.present_risk

    @property
    def climate_delta(self) -> float:
        """Risk added by the change in hazard alone."""
        return self.future_risk - self.development_risk

    def to_frame(self) -> pd.DataFrame:
        """The decomposition as a four-row table."""
        return pd.DataFrame(
            [
                {"Component": f"Risk {self.present_year}", "Value": self.present_risk},
                {"Component": "Economic development", "Value": self.development_delta},
                {"Component": "Climate change", "Value": self.climate_delta},
                {"Component": f"Risk {self.future_year}", "Value": self.future_risk},
            ]
        )


def waterfall_components(
    hazard: Hazard,
    entity: Entity,
    haz_future: Hazard,
    ent_future: Entity,
    risk_func: Callable[[Impact], float] = risk_aai_agg,
) -> WaterfallResult:
    """Split the growth in risk into economic development and climate change.

    Mirrors :py:meth:`CostBenefit.plot_waterfall` but returns the numbers rather
    than a Matplotlib axis, so the interface can draw them itself.

    Parameters
    ----------
    hazard : climada.hazard.Hazard
        Present hazard.
    entity : climada.entity.Entity
        Present entity.
    haz_future : climada.hazard.Hazard
        Future hazard.
    ent_future : climada.entity.Entity
        Future entity, whose exposures' reference year gives the horizon.
    risk_func : callable, optional
        Risk metric. Default: average annual impact.

    Returns
    -------
    WaterfallResult

    Raises
    ------
    ValueError
        If present and future entities share a reference year.
    """
    present_year = int(entity.exposures.ref_year)
    future_year = int(ent_future.exposures.ref_year)
    if present_year == future_year:
        raise ValueError(
            "Present and future entities have the same reference year "
            f"({present_year}); the waterfall needs two distinct years."
        )

    present = ImpactCalc(entity.exposures, entity.impact_funcs, hazard).impact(
        save_mat=False
    )
    development = ImpactCalc(
        ent_future.exposures, ent_future.impact_funcs, hazard
    ).impact(save_mat=False)
    future = ImpactCalc(
        ent_future.exposures, ent_future.impact_funcs, haz_future
    ).impact(save_mat=False)

    return WaterfallResult(
        present_year=present_year,
        future_year=future_year,
        present_risk=float(risk_func(present)),
        development_risk=float(risk_func(development)),
        future_risk=float(risk_func(future)),
        unit=present.unit,
    )
