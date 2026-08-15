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

Number and unit formatting shared by the CLIMADA user interface.
"""

import math
from typing import Tuple

import numpy as np

__all__ = [
    "norm_values",
    "fmt_value",
    "fmt_compact",
    "fmt_ratio",
    "fmt_percent",
    "safe_ratio",
]


def norm_values(value: float) -> Tuple[float, str]:
    """Scaling factor and suffix for a magnitude, matching CLIMADA's convention.

    Parameters
    ----------
    value : float
        Reference magnitude, e.g. the largest value on an axis.

    Returns
    -------
    tuple of (float, str)
        Divisor and its short name (``''``, ``'k'``, ``'m'`` or ``'bn'``).
    """
    value = abs(float(value)) if np.isfinite(value) else 0.0
    if value / 1.0e9 > 1:
        return 1.0e9, "bn"
    if value / 1.0e6 > 1:
        return 1.0e6, "m"
    if value / 1.0e3 > 1:
        return 1.0e3, "k"
    return 1.0, ""


def fmt_value(value: float, unit: str = "", digits: int = 2) -> str:
    """Format a monetary or physical value with a magnitude suffix.

    Parameters
    ----------
    value : float
        Value to format.
    unit : str, optional
        Unit appended after the number, e.g. ``'USD'``. Default: ``''``.
    digits : int, optional
        Decimal places. Default: 2.

    Returns
    -------
    str
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    factor, name = norm_values(value)
    number = f"{value / factor:,.{digits}f}"
    suffix = " ".join(part for part in (name, unit) if part)
    return f"{number} {suffix}".strip()


def fmt_compact(value: float, unit: str = "") -> str:
    """Format a value for a stat tile: no decimals above 100, one below.

    Parameters
    ----------
    value : float
        Value to format.
    unit : str, optional
        Unit appended after the number. Default: ``''``.

    Returns
    -------
    str
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    factor, name = norm_values(value)
    scaled = value / factor
    digits = 0 if abs(scaled) >= 100 else 1
    suffix = " ".join(part for part in (name, unit) if part)
    return f"{scaled:,.{digits}f} {suffix}".strip()


def fmt_ratio(value: float, digits: int = 2) -> str:
    """Format a dimensionless ratio, guarding against infinities.

    Parameters
    ----------
    value : float
        Ratio to format.
    digits : int, optional
        Decimal places. Default: 2.

    Returns
    -------
    str
    """
    if value is None or isinstance(value, str):
        return "n/a"
    if not np.isfinite(value):
        return "0.00" if value == -math.inf else "n/a"
    return f"{value:,.{digits}f}"


def fmt_percent(value: float, digits: int = 1) -> str:
    """Format a fraction as a percentage.

    Parameters
    ----------
    value : float
        Fraction, e.g. 0.031 for 3.1%.
    digits : int, optional
        Decimal places. Default: 1.

    Returns
    -------
    str
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:,.{digits}f}%"


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, returning NaN instead of raising when the denominator vanishes.

    Parameters
    ----------
    numerator : float
    denominator : float

    Returns
    -------
    float
        ``numerator / denominator``, or NaN if that is undefined.
    """
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    result = numerator / denominator
    return result if np.isfinite(result) else float("nan")
