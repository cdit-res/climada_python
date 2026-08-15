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

Reusable Streamlit pieces for the CLIMADA user interface.
"""

import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from climada.ui import state

__all__ = [
    "chart_with_table",
    "download_frame",
    "page_header",
    "requires_inputs",
    "save_upload",
    "stat_row",
]


def page_header(title: str, lead: str) -> None:
    """Page title and one-paragraph orientation.

    Parameters
    ----------
    title : str
        Page heading.
    lead : str
        Sentence explaining what the page is for.
    """
    st.title(title)
    st.caption(lead)


def stat_row(stats: Sequence[Tuple[str, str, Optional[str]]]) -> None:
    """A row of stat tiles.

    Parameters
    ----------
    stats : sequence of (label, value, help) tuples
        ``help`` may be ``None``.
    """
    columns = st.columns(len(stats))
    for column, (label, value, tooltip) in zip(columns, stats):
        with column:
            st.metric(label=label, value=value, help=tooltip)


def chart_with_table(
    figure: go.Figure,
    frame: Optional[pd.DataFrame] = None,
    table_label: str = "Show the numbers",
    download_name: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """Render a chart with its underlying table one click away.

    The table is not decoration: the lighter categorical steps used on the
    light surface are only legible with a text alternative present, and a
    reader who needs exact figures should never have to hover over marks.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
    frame : pandas.DataFrame, optional
        Data behind the chart. Omit only for charts with no tabular form.
    table_label : str, optional
        Label of the expander holding the table.
    download_name : str, optional
        File name offered for CSV download. Default: no download button.
    key : str, optional
        Streamlit key, needed when several charts share a page.
    """
    st.plotly_chart(figure, use_container_width=True, key=key)
    if frame is None or frame.empty:
        return
    with st.expander(table_label):
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if download_name:
            download_frame(frame, download_name, key=f"{key or table_label}-dl")


def download_frame(
    frame: pd.DataFrame, file_name: str, key: Optional[str] = None
) -> None:
    """A CSV download button for a table.

    Parameters
    ----------
    frame : pandas.DataFrame
    file_name : str
        Name offered to the browser.
    key : str, optional
        Streamlit key.
    """
    st.download_button(
        "Download CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def requires_inputs(*, needs_measures: bool = False) -> bool:
    """Guard a page that cannot run until data is loaded.

    Renders an explanation and returns ``False`` when something is missing, so
    callers can ``if not requires_inputs(): return``.

    Parameters
    ----------
    needs_measures : bool, optional
        Also require at least one adaptation measure. Default: False.

    Returns
    -------
    bool
        Whether the page has everything it needs.
    """
    missing: List[str] = []
    if state.get("hazard") is None:
        missing.append("a hazard event set")
    if state.get("exposures") is None:
        missing.append("exposures")
    if state.get("impf_set") is None:
        missing.append("impact functions")
    if needs_measures and not state.measure_rows():
        missing.append("at least one adaptation measure")

    if not missing:
        return True

    st.info(
        "This page needs "
        + _join_clauses(missing)
        + ". "
        + (
            "Define them on the **Measures** page."
            if missing == ["at least one adaptation measure"]
            else "Load them on the **Data** page -- the quickest route is the "
            "bundled demo scenario."
        )
    )
    return False


def _join_clauses(items: Iterable[str]) -> str:
    """Join clauses into readable prose: 'a, b and c'."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def save_upload(uploaded_file, directory: Path) -> Path:
    """Persist an uploaded file so CLIMADA's file readers can open it.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The object returned by ``st.file_uploader``.
    directory : pathlib.Path
        Directory to write into; created if absent.

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def error_box(err: Exception, what: str) -> None:
    """Report a failure without dumping a traceback into the page.

    Parameters
    ----------
    err : Exception
        The failure.
    what : str
        What was being attempted, e.g. ``'load the hazard'``.
    """
    st.error(f"Could not {what}: {err}")
    with st.expander("Technical detail"):
        st.exception(err)


def excel_bytes(frames: Dict[str, pd.DataFrame]) -> bytes:
    """Pack several tables into one Excel workbook.

    Parameters
    ----------
    frames : dict
        Mapping of sheet name to table. Sheet names are truncated to Excel's
        31-character limit.

    Returns
    -------
    bytes
        The workbook.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()
