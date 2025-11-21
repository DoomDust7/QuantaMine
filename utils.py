# analytics/utils.py
import math
import numpy as np
import pandas as pd
from typing import Any

def safe_ratio(a, b):
    try:
        if b == 0 or pd.isna(a) or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _mean_non_nan(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) > 0 else float("nan")

def clamp01(x):
    if pd.isna(x):
        return 0.5
    return float(min(max(x, 0.0), 1.0))

def scale_0_5(x):
    if pd.isna(x):
        return 2.5
    return float(min(max(x * 5.0, 0.0), 5.0))
