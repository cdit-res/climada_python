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

Entry point of the CLIMADA Streamlit interface.

Run it with ``climada-ui``, ``python -m climada.ui``, or directly with
``streamlit run climada/ui/app.py``.
"""

import logging

import streamlit as st

from climada.ui import state
from climada.ui.views import cba_view, data_view, export_view, measures_view, risk_view

LOGGER = logging.getLogger(__name__)


def _sidebar() -> None:
    """Persistent status panel and display options."""
    with st.sidebar:
        st.markdown("### Current analysis")
        summary = state.state_summary()
        for name, description in summary.items():
            icon = "✓" if description else "·"
            st.markdown(
                f"{icon} **{name.capitalize()}**  \n"
                f"<span style='opacity:0.7'>{description or 'not loaded'}</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### Display")
        st.checkbox(
            "Map basemap",
            value=bool(state.get("basemap")),
            key="basemap",
            help="Draws OpenStreetMap tiles behind the maps. Needs the browser "
            "to reach the tile server.",
        )
        st.number_input(
            "Maximum points on a map",
            min_value=500,
            max_value=100_000,
            value=int(state.get("map_point_limit")),
            step=500,
            key="map_point_limit",
            help="Large exposure sets are thinned to the highest-value points "
            "so the map stays responsive.",
        )

        st.divider()
        if st.button("Clear results", help="Keeps the loaded data, drops the numbers"):
            state.invalidate()
            st.rerun()

        st.caption(
            "Built on CLIMADA. Physical risk from `ImpactCalc`, appraisal from "
            "`CostBenefit` -- no bespoke maths in this interface."
        )


def main() -> None:
    """Configure the page, build the navigation and run the selected view."""
    st.set_page_config(
        page_title="CLIMADA risk and adaptation",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    state.ensure_defaults()
    _sidebar()

    pages = [
        st.Page(
            data_view.render, title="Data", icon=":material/database:", url_path="data"
        ),
        st.Page(
            risk_view.render, title="Risk", icon=":material/flood:", url_path="risk"
        ),
        st.Page(
            measures_view.render,
            title="Measures",
            icon=":material/construction:",
            url_path="measures",
        ),
        st.Page(
            cba_view.render,
            title="Cost-benefit",
            icon=":material/balance:",
            url_path="cost-benefit",
        ),
        st.Page(
            export_view.render,
            title="Report",
            icon=":material/description:",
            url_path="report",
        ),
    ]
    st.navigation(pages, position="top").run()


main()
