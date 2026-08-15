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

Test the CLIMADA user interface's analysis, chart and report layers.

The Streamlit view modules are not exercised here; everything the interface
computes lives in modules that do not import Streamlit, and that is what these
tests cover.
"""

import math
import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd

PLOTLY = find_spec("plotly") is not None

from climada.ui import analysis, datasets, formatting, report
from climada.ui.formatting import norm_values, safe_ratio

if PLOTLY:
    from climada.ui import charts, theme


class TestFormatting(unittest.TestCase):
    """Number and unit formatting."""

    def test_norm_values(self):
        """Magnitude suffixes follow CLIMADA's thresholds."""
        self.assertEqual(norm_values(5.0), (1.0, ""))
        self.assertEqual(norm_values(5.0e3), (1.0e3, "k"))
        self.assertEqual(norm_values(5.0e6), (1.0e6, "m"))
        self.assertEqual(norm_values(5.0e9), (1.0e9, "bn"))

    def test_norm_values_handles_nan(self):
        """A non-finite reference falls back to no scaling."""
        self.assertEqual(norm_values(float("nan")), (1.0, ""))

    def test_fmt_value(self):
        """Values carry their magnitude and unit."""
        self.assertEqual(formatting.fmt_value(1.5e9, "USD"), "1.50 bn USD")
        self.assertEqual(formatting.fmt_value(float("nan"), "USD"), "n/a")

    def test_fmt_compact_precision(self):
        """Stat tiles drop the decimal above 100."""
        self.assertEqual(formatting.fmt_compact(1.234e9, "USD"), "1.2 bn USD")
        self.assertEqual(formatting.fmt_compact(1.234e11, "USD"), "123 bn USD")

    def test_fmt_percent_and_ratio(self):
        """Percentages and ratios round as documented."""
        self.assertEqual(formatting.fmt_percent(0.0314, digits=1), "3.1%")
        self.assertEqual(formatting.fmt_ratio(1.239), "1.24")
        self.assertEqual(formatting.fmt_ratio(float("nan")), "n/a")

    def test_safe_ratio(self):
        """Division by zero yields NaN rather than raising."""
        self.assertEqual(safe_ratio(4.0, 2.0), 2.0)
        self.assertTrue(math.isnan(safe_ratio(1.0, 0.0)))
        self.assertTrue(math.isnan(safe_ratio(1.0, float("nan"))))


@unittest.skipUnless(PLOTLY, "plotly is not installed")
class TestTheme(unittest.TestCase):
    """Palette and Plotly template."""

    def test_palettes_are_parallel(self):
        """Light and dark carry the same number of slots."""
        self.assertEqual(len(theme.categorical(False)), len(theme.categorical(True)))
        self.assertEqual(len(theme.categorical()), theme.MAX_CATEGORICAL_SERIES)

    def test_sequential_orientation(self):
        """The dark ramp is the light one reversed, so near-zero recedes."""
        self.assertEqual(
            theme.sequential_scale(True), list(reversed(theme.sequential_scale(False)))
        )

    def test_template_paints_a_surface(self):
        """Backgrounds are explicit, never inherited from the host page."""
        for dark in (False, True):
            template = theme.plotly_template(dark)
            self.assertEqual(
                template.layout.paper_bgcolor, theme.tokens(dark)["surface"]
            )
            self.assertEqual(
                template.layout.plot_bgcolor, theme.tokens(dark)["surface"]
            )

    def test_status_colours_are_not_series_colours(self):
        """A status colour must never impersonate a categorical slot."""
        for value in theme.STATUS.values():
            self.assertNotIn(value, theme.categorical(False))
            self.assertNotIn(value, theme.categorical(True))


class TestMeasureSpecs(unittest.TestCase):
    """Building measures from edited rows."""

    def test_default_row_is_inert(self):
        """A fresh row leaves the vulnerability curve untouched."""
        measure = analysis.build_measure(analysis.default_measure_row(), "TC")
        self.assertEqual(measure.hazard_inten_imp, (1.0, 0.0))
        self.assertEqual(measure.mdd_impact, (1.0, 0.0))
        self.assertEqual(measure.paa_impact, (1.0, 0.0))
        self.assertEqual(measure.cost, 0.0)

    def test_intensity_offset_sign(self):
        """A negative offset shifts the curve right, which is protection."""
        row = analysis.default_measure_row("wall")
        row["hazard_inten_imp_b"] = -4.0
        measure = analysis.build_measure(row, "TC")
        self.assertEqual(measure.hazard_inten_imp, (1.0, -4.0))

    def test_duplicate_names_rejected(self):
        """CLIMADA keys measures by name, so duplicates must not slip through."""
        rows = [
            analysis.default_measure_row("same"),
            analysis.default_measure_row("same"),
        ]
        with self.assertRaises(ValueError) as raised:
            analysis.build_measure_set(rows, "TC")
        self.assertIn("unique", str(raised.exception))

    def test_blank_name_rejected(self):
        """An unnamed measure cannot be addressed in the results."""
        row = analysis.default_measure_row("")
        with self.assertRaises(ValueError):
            analysis.build_measure_set([row], "TC")

    def test_measure_set_round_trip(self):
        """Every row becomes a measure of the right hazard type."""
        rows = [
            analysis.default_measure_row("a"),
            analysis.default_measure_row("b"),
        ]
        measure_set = analysis.build_measure_set(rows, "TC")
        self.assertEqual(measure_set.size(), 2)
        self.assertEqual(sorted(measure_set.get_names("TC")), ["a", "b"])


class TestDiscRates(unittest.TestCase):
    """Discount rate construction."""

    def test_constant_rate_covers_horizon(self):
        """Every year between present and horizon carries the rate."""
        disc = analysis.build_disc_rates(0.03, 2025, 2030)
        self.assertEqual(list(disc.years), list(range(2025, 2031)))
        np.testing.assert_allclose(disc.rates, 0.03)

    def test_from_table_sorts_and_drops_blanks(self):
        """An edited table may arrive unsorted and with empty rows."""
        frame = pd.DataFrame(
            {"year": [2027, 2025, None, 2026], "rate": [0.03, 0.01, 0.02, None]}
        )
        disc = analysis.disc_rates_from_table(frame)
        self.assertEqual(list(disc.years), [2025, 2027])


class TestBenefitCostRatio(unittest.TestCase):
    """Inverting CLIMADA's cost/benefit ratio."""

    def test_zero_benefit_gives_zero_ratio(self):
        """A measure that averts nothing scores zero, not 'undefined'."""
        self.assertEqual(analysis._benefit_cost_ratio(float("inf")), 0.0)

    def test_zero_cost_gives_infinite_ratio(self):
        """A costless measure with a benefit is unboundedly efficient."""
        self.assertEqual(analysis._benefit_cost_ratio(0.0), float("inf"))

    def test_normal_ratio_is_inverted(self):
        """The ordinary case is a plain reciprocal."""
        self.assertAlmostEqual(analysis._benefit_cost_ratio(0.5), 2.0)

    def test_nan_stays_nan(self):
        """An undefined ratio must not be reported as a number."""
        self.assertTrue(math.isnan(analysis._benefit_cost_ratio(float("nan"))))


class TestExposureTable(unittest.TestCase):
    """Building exposures from a plain table."""

    def test_from_table(self):
        """Coordinates, values and an impact function column come through."""
        frame = pd.DataFrame(
            {"lat": [26.0, 25.5], "lon": [-80.0, -80.5], "value": [1.0e6, 2.0e6]}
        )
        exposures = datasets.exposures_from_table(
            frame, value_unit="EUR", ref_year=2030, haz_type="TC"
        )
        self.assertEqual(len(exposures.gdf), 2)
        self.assertEqual(exposures.value_unit, "EUR")
        self.assertEqual(exposures.ref_year, 2030)
        self.assertIn("impf_TC", exposures.gdf.columns)

    def test_missing_columns_rejected(self):
        """A table without coordinates cannot become exposures."""
        with self.assertRaises(ValueError) as raised:
            datasets.exposures_from_table(pd.DataFrame({"value": [1.0]}))
        self.assertIn("latitude", str(raised.exception))


class TestImpactFunctions(unittest.TestCase):
    """Vulnerability curve helpers."""

    def test_step_impf(self):
        """A step curve is registered under the requested hazard type."""
        impf_set = datasets.step_impf_set("RF", intensity_high=2.0, damage_ratio=0.6)
        curves = datasets.impf_curves(impf_set, "RF")
        self.assertEqual(len(curves), 1)
        self.assertAlmostEqual(max(next(iter(curves.values()))["mdr"]), 0.6)

    def test_emanuel_impf(self):
        """The Emanuel curve rises monotonically above the threshold."""
        impf_set = datasets.emanuel_impf_set(v_thresh=25.0, v_half=60.0)
        mdr = next(iter(datasets.impf_curves(impf_set, "TC").values()))["mdr"]
        self.assertTrue(np.all(np.diff(mdr) >= -1e-12))

    def test_default_for_unknown_hazard_warns_in_note(self):
        """An unsupported hazard type gets a placeholder and says so."""
        _, note = datasets.default_impf_set("XYZ")
        self.assertIn("placeholder", note.lower())

    def test_impf_curves_without_haz_type(self):
        """Asking for every curve flattens CLIMADA's nested mapping."""
        impf_set = datasets.step_impf_set("RF")
        self.assertEqual(len(datasets.impf_curves(impf_set)), 1)


class TestReport(unittest.TestCase):
    """Reproduction script and assumption tables."""

    LABELS = {
        "hazard": "demo",
        "exposures": "demo",
        "impf": "demo",
        "measures": "1 measure",
    }
    META = {
        "present_year": 2025,
        "future_year": 2050,
        "disc_rate": 0.02,
        "risk_metric": "Average annual impact",
        "imp_time_depen": 1.0,
        "scenario": True,
        "growth_factor": 1.3,
        "intensity_factor": 1.1,
        "frequency_factor": 1.0,
    }

    def test_script_compiles(self):
        """The exported script must be valid Python."""
        source = report.reproduction_script(
            self.META,
            self.LABELS,
            [analysis.default_measure_row("wall")],
            "TC",
            demo_key="tc_florida",
        )
        compile(source, "reproduction", "exec")
        self.assertIn("run_cost_benefit", source)
        self.assertIn("tc_florida", source)

    def test_script_without_demo_names_the_files(self):
        """Without a demo the script tells the reader what to point it at."""
        source = report.reproduction_script(
            self.META, self.LABELS, [], "TC", demo_key=None
        )
        compile(source, "reproduction", "exec")
        self.assertIn("Hazard.from_hdf5", source)

    def test_assumptions_include_the_scenario(self):
        """Scenario factors must appear when a scenario was modelled."""
        frame = report.assumptions_table(self.META, self.LABELS)
        settings = list(frame["Setting"])
        self.assertIn("Discount rate", settings)
        self.assertIn("Hazard intensity factor", settings)

    def test_assumptions_omit_factors_without_scenario(self):
        """A static run should not imply factors it never applied."""
        meta = dict(self.META, scenario=False)
        settings = list(report.assumptions_table(meta, self.LABELS)["Setting"])
        self.assertNotIn("Hazard intensity factor", settings)


@unittest.skipUnless(
    datasets.DEMO_SCENARIOS["tc_florida"].available, "demo data not installed"
)
class TestEndToEnd(unittest.TestCase):
    """The full workflow against CLIMADA's bundled demo data."""

    @classmethod
    def setUpClass(cls):
        bundle = datasets.load_demo("tc_florida")
        cls.hazard = bundle["hazard"]
        cls.exposures = bundle["entity"].exposures
        cls.impf_set = bundle["entity"].impact_funcs
        cls.risk = analysis.compute_risk(cls.exposures, cls.impf_set, cls.hazard)

    def test_risk_result(self):
        """The assessment reports a positive, plausible annual loss."""
        self.assertGreater(self.risk.aai, 0)
        self.assertGreater(self.risk.total_value, self.risk.aai)
        self.assertEqual(self.risk.haz_type, "TC")
        self.assertGreater(self.risk.rp_value(250), self.risk.rp_value(50))

    def test_loss_ratio(self):
        """The loss ratio is the annual loss over the exposed value."""
        self.assertAlmostEqual(
            self.risk.loss_ratio, self.risk.aai / self.risk.total_value
        )

    def test_exceedance_table(self):
        """One row per requested return period, losses rising with period."""
        frame = analysis.exceedance_table(self.risk)
        self.assertEqual(len(frame), len(analysis.DEFAULT_RETURN_PERIODS))
        losses = frame[f"Impact ({self.risk.unit})"].values
        self.assertTrue(np.all(np.diff(losses) >= -1e-6))

    def test_top_events(self):
        """The event table is sorted worst first."""
        frame = analysis.top_events_table(self.risk.impact, limit=5)
        self.assertEqual(len(frame), 5)
        impacts = frame[f"Impact ({self.risk.impact.unit})"].values
        self.assertTrue(np.all(np.diff(impacts) <= 1e-6))

    def test_impact_points_limited(self):
        """The map table honours its point cap and stays sorted."""
        frame = analysis.impact_point_table(self.risk.impact, limit=10)
        self.assertLessEqual(len(frame), 10)
        self.assertTrue(np.all(np.diff(frame["eai"].values) <= 1e-6))

    def test_protection_reduces_risk(self):
        """A negative intensity offset must lower the annual loss."""
        row = analysis.default_measure_row("wall")
        row["hazard_inten_imp_b"] = -4.0
        measure = analysis.build_measure(row, "TC")
        new_exp, new_impf, new_haz = analysis.measure_impact_curves(
            measure, self.exposures, self.impf_set, self.hazard
        )
        treated = analysis.compute_risk(new_exp, new_impf, new_haz)
        self.assertLess(treated.aai, self.risk.aai)

    def test_scale_hazard_leaves_original_alone(self):
        """Scenario building must not mutate the loaded hazard."""
        before = self.hazard.intensity.max()
        scaled = analysis.scale_hazard(self.hazard, 1.5, 2.0)
        self.assertAlmostEqual(self.hazard.intensity.max(), before)
        self.assertAlmostEqual(scaled.intensity.max(), before * 1.5, places=3)
        np.testing.assert_allclose(scaled.frequency, self.hazard.frequency * 2.0)

    def test_grow_exposures_leaves_original_alone(self):
        """Likewise for exposure growth."""
        before = float(np.nansum(self.exposures.value))
        grown = analysis.grow_exposures(self.exposures, 1.4, ref_year=2050)
        self.assertAlmostEqual(float(np.nansum(self.exposures.value)), before)
        self.assertAlmostEqual(float(np.nansum(grown.value)), before * 1.4, places=0)
        self.assertEqual(grown.ref_year, 2050)

    def test_cost_benefit(self):
        """A costed measure is ranked, and the table is internally consistent."""
        rows = [analysis.default_measure_row("wall")]
        rows[0]["hazard_inten_imp_b"] = -4.0
        rows[0]["cost"] = 1.0e9
        measure_set = analysis.build_measure_set(rows, "TC")
        disc = analysis.build_disc_rates(0.02, 2018, 2040)
        entity = analysis.build_entity(
            self.exposures, self.impf_set, measure_set, disc, ref_year=2018
        )
        ent_future = analysis.build_entity(
            analysis.grow_exposures(self.exposures, 1.2, ref_year=2040),
            self.impf_set,
            measure_set,
            disc,
            ref_year=2040,
        )
        cost_ben = analysis.run_cost_benefit(
            self.hazard,
            entity,
            haz_future=analysis.scale_hazard(self.hazard, 1.1),
            ent_future=ent_future,
            future_year=2040,
            imp_time_depen=1.0,
        )

        table = analysis.cost_benefit_table(cost_ben)
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertGreater(row[f"Benefit ({cost_ben.unit})"], 0)
        self.assertAlmostEqual(
            row["Benefit/cost ratio"],
            row[f"Benefit ({cost_ben.unit})"] / row[f"Cost ({cost_ben.unit})"],
            places=6,
        )

        summary = analysis.cost_benefit_summary(cost_ben)
        self.assertEqual(summary["present_year"], 2018)
        self.assertEqual(summary["future_year"], 2040)
        self.assertGreater(summary["total_climate_risk"], 0)
        self.assertGreater(
            summary["annual_risk_future"], summary["annual_risk_present"]
        )

    def test_cost_benefit_rejects_a_backwards_horizon(self):
        """A horizon that does not advance would make every benefit undefined."""
        measure_set = analysis.build_measure_set(
            [analysis.default_measure_row("wall")], "TC"
        )
        disc = analysis.build_disc_rates(0.02, 2018, 2018)
        entity = analysis.build_entity(
            self.exposures, self.impf_set, measure_set, disc, ref_year=2018
        )
        ent_future = analysis.build_entity(
            self.exposures, self.impf_set, measure_set, disc, ref_year=2018
        )
        with self.assertRaises(ValueError) as raised:
            analysis.run_cost_benefit(
                self.hazard, entity, ent_future=ent_future, future_year=2018
            )
        self.assertIn("must end after", str(raised.exception))

    def test_waterfall_splits_the_growth(self):
        """Development and climate deltas sum to the total change in risk."""
        measure_set = analysis.build_measure_set(
            [analysis.default_measure_row("wall")], "TC"
        )
        disc = analysis.build_disc_rates(0.02, 2018, 2040)
        entity = analysis.build_entity(
            self.exposures, self.impf_set, measure_set, disc, ref_year=2018
        )
        ent_future = analysis.build_entity(
            analysis.grow_exposures(self.exposures, 1.5, ref_year=2040),
            self.impf_set,
            measure_set,
            disc,
            ref_year=2040,
        )
        result = analysis.waterfall_components(
            self.hazard, entity, analysis.scale_hazard(self.hazard, 1.2), ent_future
        )
        self.assertAlmostEqual(
            result.present_risk + result.development_delta + result.climate_delta,
            result.future_risk,
            places=3,
        )
        self.assertGreater(result.development_delta, 0)
        self.assertGreater(result.climate_delta, 0)
        self.assertEqual(len(result.to_frame()), 4)

    def test_waterfall_needs_two_years(self):
        """Identical reference years make the decomposition meaningless."""
        measure_set = analysis.build_measure_set(
            [analysis.default_measure_row("wall")], "TC"
        )
        disc = analysis.build_disc_rates(0.02, 2018, 2040)
        entity = analysis.build_entity(
            self.exposures, self.impf_set, measure_set, disc, ref_year=2018
        )
        with self.assertRaises(ValueError):
            analysis.waterfall_components(self.hazard, entity, self.hazard, entity)

    @unittest.skipUnless(PLOTLY, "plotly is not installed")
    def test_every_chart_renders(self):
        """Each figure serialises, in both light and dark mode."""
        unit = self.risk.unit
        points = analysis.impact_point_table(self.risk.impact, limit=50)
        for dark in (False, True):
            figures = [
                charts.exceedance_curve(
                    {
                        "base": (
                            self.risk.freq_curve_return_per,
                            self.risk.freq_curve_impact,
                        )
                    },
                    unit,
                    dark=dark,
                ),
                charts.return_period_bars(
                    self.risk.return_periods, self.risk.rp_impact, unit, dark=dark
                ),
                charts.series_bar_chart(["a", "b"], [1.0, 2.0], unit, dark=dark),
                charts.point_map(points, "eai", unit, dark=dark),
                charts.impact_function_chart(
                    datasets.impf_curves(self.impf_set, "TC"), "m/s", dark=dark
                ),
                charts.cost_benefit_chart(
                    ["a", "b"], [2.0e9, 1.0e9], [1.4, 0.6], 4.0e9, unit, 25, dark=dark
                ),
                charts.measure_bcr_bars(["a", "b"], [1.4, 0.6], dark=dark),
                charts.residual_risk_bars(["a"], [1.0e6], [2.0e6], unit, dark=dark),
                charts.waterfall_chart(
                    2018, 2040, 1.0e8, 3.0e7, 5.0e7, unit, dark=dark
                ),
                charts.discount_rate_chart([2018, 2019], [0.02, 0.02], dark=dark),
            ]
            for figure in figures:
                self.assertTrue(figure.to_json())

    @unittest.skipUnless(PLOTLY, "plotly is not installed")
    def test_empty_map_does_not_raise(self):
        """An empty point table renders a message rather than failing."""
        figure = charts.point_map(
            pd.DataFrame(columns=["latitude", "longitude", "eai"]), "eai"
        )
        self.assertTrue(figure.to_json())


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFormatting)
    for case in (
        TestTheme,
        TestMeasureSpecs,
        TestDiscRates,
        TestBenefitCostRatio,
        TestExposureTable,
        TestImpactFunctions,
        TestReport,
        TestEndToEnd,
    ):
        TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(case))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
