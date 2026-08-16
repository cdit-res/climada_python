# CLIMADA user interface

A browser interface for the two things CLIMADA is most often used for:
**physical climate risk assessment** and **adaptation cost-benefit analysis**.

It is a shell around `climada.engine.ImpactCalc` and
`climada.engine.CostBenefit` — no modelling happens in the interface itself.

## Install and run

```bash
pip install climada[ui]
climada-ui
```

Equivalent entry points, if you prefer them:

```bash
python -m climada.ui
streamlit run climada/ui/app.py
```

The interface opens at <http://localhost:8501>. `climada-ui --help` lists the
port, address and headless options.

## The five pages

| Page | What it does |
|---|---|
| **Data** | Load the hazard, the exposures and the vulnerability curves — from the bundled demo, from files, or from the CLIMADA Data API. Checks that the three fit together before you run anything. |
| **Risk** | Average annual impact, loss exceedance curve, losses at chosen return periods, a map of where the risk sits, and the worst events in the set. |
| **Measures** | An editable table of adaptation options. Presets for coastal protection, retrofit, exposure reduction and insurance. Preview what a measure does to the vulnerability curve and to the annual loss before committing to it. |
| **Cost-benefit** | Discounts averted damage over the appraisal horizon and ranks the measures. Includes the classic CLIMADA cost-benefit chart, a waterfall splitting risk growth into economic development and climate change, residual-risk curves, measure combination and risk-transfer layering. |
| **Report** | A plain-language summary, the assumptions behind it, every table as CSV or one Excel workbook, and a Python script that reproduces the run. |

## Fastest route to a result

1. **Data** → *Start from a demo* → **Load demo scenario**. That brings in
   CLIMADA's Florida tropical cyclone set, matching exposures, curves, four
   coastal measures and a discount rate.
2. **Risk** → **Run risk assessment**.
3. **Cost-benefit** → set the horizon and scenario → **Run cost-benefit**.
4. **Report** → download the workbook, or the script.

## Heat risk

CLIMADA's core ships no heat hazard class and no heat vulnerability curve, the
Data API serves no heat dataset, and Petals 6.2 has no heat module either — its
hazards are flood, drought, landslide, low-flow, crop yield, TC rainfield and
surge, and wildfire. What the documentation does give is the framing: *"the
Proportion of Assets Affected gives the fraction of exposures that are
affected, such as the mortality rate in a population from a heatwave."*
`climada.ui.heat` supplies the missing machinery on that framing.

**Hazard.** Bring gridded daily temperature — ERA5, CMIP6, or any NetCDF/GRIB
xarray can open — via the *Gridded temperature (heat)* tab, which reads it into
a `Hazard` of type `HW` (CLIMADA's EM-DAT code for extreme temperature). Every
time step becomes one event.

**The frequency trap.** A daily record sampled over *n* years must give each
day-event a frequency of `1/n` per year; leaving it at CLIMADA's default of 1.0
inflates the annual impact by the length of the record. The loader sets this
for you, counting the distinct calendar years the events fall in — which is
what makes warm-season data work, since twenty Junes-to-Septembers sample
twenty years even though the first and last day are only 19.3 years apart.
Override it with the record-length box when that reading is wrong.

**What to count.** Four metrics, all built from population exposure:

| Metric | Impact unit | Parameters |
|---|---|---|
| Heat-attributable mortality | deaths | minimum-mortality temperature, extra risk per °C, baseline death rate |
| Person-days above a threshold | person-days | threshold |
| Person-degree-days | person-degree-days | base temperature |
| Lost labour | lost work-days | temperature where productivity falls, where work stops |

The interface relabels itself accordingly: a mortality run reports annual
deaths and deaths per 100,000, not "loss" and "loss ratio".

**Exposure.** The LitPop tab has a population layer (`fin_mode='pop'`), giving
gridded people per country from the Data API.

**Measures.** Heat presets replace the coastal ones when an `HW` hazard is
loaded: urban greening and cool roofs as negative intensity offsets, warning
systems, cooling centres and home retrofit as reductions in the harm rate.

**Try it without data.** *Data → Start from a demo → Heat (generated)* builds a
seeded synthetic city: 20 warm seasons of daily temperature over a 25 km grid
with an urban heat island, a warming trend, and a population concentrated where
it is hottest. It is a statistical toy for learning the workflow, clearly
labelled as such — not a climatology, and not a basis for any statement about a
real place.

**The curves are screening relationships, not epidemiology.** Real
exposure-response functions are non-linear, lagged over several days, and
differ by age, city and acclimatisation. Calibrate the minimum-mortality
temperature and the risk per degree to local literature before reporting death
counts.

## Using the analysis layer without the interface

`climada.ui.analysis`, `climada.ui.charts`, `climada.ui.datasets` and
`climada.ui.report` import no Streamlit, so they work from a script or a
notebook. The **Report** page emits exactly this kind of code:

```python
from climada.ui.analysis import (
    build_disc_rates, build_entity, build_measure_set,
    compute_risk, cost_benefit_table, run_cost_benefit,
)
from climada.ui.datasets import load_demo

bundle = load_demo("tc_florida")
hazard, entity = bundle["hazard"], bundle["entity"]

risk = compute_risk(entity.exposures, entity.impact_funcs, hazard)
print(f"{risk.aai:,.0f} {risk.unit} per year, "
      f"{risk.rp_value(100):,.0f} at the 100-year return period")
```

## Notes on the modelling

- **Measure costs are entered already discounted**, as CLIMADA expects. The
  interface discounts the stream of *averted damage*, not the costs.
- **The intensity offset is signed**: a *negative* offset shifts the
  vulnerability curve to the right and is therefore protection. `-4` on a
  wind hazard means the assets behave as though the wind were 4 m/s weaker.
- **Uniform hazard scaling factors are a screening device.** Multiplying every
  event's intensity by 1.1 is not a climate projection; for a defensible study,
  load a future hazard set from the Data API instead.
- **Benefits are appraised one measure at a time**, so they overlap. Use
  *Combine measures* on the cost-benefit page for the joint figure.

## Layout

```
climada/ui/
├── app.py          Streamlit entry point and navigation
├── cli.py          climada-ui launcher
├── analysis.py     every computation, no Streamlit
├── heat.py         heat hazard, curves and measures, no Streamlit
├── charts.py       Plotly figures, no Streamlit
├── datasets.py     loading hazard/exposures/curves, no Streamlit
├── report.py       exports and the reproduction script, no Streamlit
├── theme.py        validated colour palette and Plotly template
├── formatting.py   number and unit formatting
├── components.py   shared Streamlit widgets
├── state.py        session state and invalidation
└── views/          one module per page
```
