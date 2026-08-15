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
├── charts.py       Plotly figures, no Streamlit
├── datasets.py     loading hazard/exposures/curves, no Streamlit
├── report.py       exports and the reproduction script, no Streamlit
├── theme.py        validated colour palette and Plotly template
├── formatting.py   number and unit formatting
├── components.py   shared Streamlit widgets
├── state.py        session state and invalidation
└── views/          one module per page
```
