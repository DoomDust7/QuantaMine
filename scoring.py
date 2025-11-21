# analytics/scoring.py
import numpy as np
import pandas as pd
from .utils import _mean_non_nan

# Basic absolute scoring functions (0-5) with nicer fallbacks

def score_linear(value, low_good, low_bad, high_bad, high_good):
    """
    Generic linear mapping: value in [low_bad..high_bad] maps to [0..5],
    with clipping and special handling when value is missing (returns NaN)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    # If lower is better
    if low_good < low_bad:
        # e.g. for metrics where very low is great and high is bad; invert if needed
        pass
    # We'll handle common metrics individually elsewhere
    return np.nan

def score_value_from_pe(pe):
    if pe is None or np.isnan(pe) or pe <= 0:
        return 2.5
    if pe <= 5:
        return 5.0
    if pe >= 40:
        return 0.0
    return round(max(0.0, 5.0 - (pe - 5) * (5.0 / (40 - 5))), 2)

def score_pb(pb):
    if pb is None or np.isnan(pb) or pb <= 0:
        return 2.5
    if pb <= 1:
        return 5.0
    if pb >= 6:
        return 0.0
    return round(max(0.0, 5.0 - (pb - 1) * (5.0 / (6 - 1))), 2)

def score_peg(peg):
    if peg is None or np.isnan(peg) or peg <= 0:
        return 2.5
    if peg <= 0.5:
        return 5.0
    if peg >= 3:
        return 0.0
    return round(max(0.0, 5.0 - (peg - 0.5) * (5.0 / (3 - 0.5))), 2)

def score_div_yield(div):
    if div is None or np.isnan(div) or div <= 0:
        return 0.0
    return round(min(div / 0.06 * 5.0, 5.0), 2)

def score_ev_ebitda(ev_ebitda):
    if ev_ebitda is None or np.isnan(ev_ebitda) or ev_ebitda <= 0:
        return 2.5
    if ev_ebitda <= 5:
        return 5.0
    if ev_ebitda >= 40:
        return 0.0
    return round(max(0.0, 5.0 - (ev_ebitda - 5) * (5.0 / (40 - 5))), 2)

def score_roe(roe):
    if roe is None or np.isnan(roe):
        return 2.5
    return round(min(max(roe / 0.20 * 5.0, 0.0), 5.0), 2)

def score_rev_growth(g):
    if g is None or np.isnan(g):
        return 2.5
    return round(min(max(g / 0.2 * 5.0, 0.0), 5.0), 2)

def score_eps_growth(g):
    if g is None or np.isnan(g):
        return 2.5
    return round(min(max(g / 0.2 * 5.0, 0.0), 5.0), 2)

def score_fcf_yield(fcy):
    if fcy is None or np.isnan(fcy):
        return 2.5
    return round(min(max(fcy / 0.05 * 5.0, 0.0), 5.0), 2)

def score_debt_to_equity(dte):
    if dte is None or np.isnan(dte):
        return 2.5
    if dte <= 0:
        return 5.0
    if dte >= 2:
        return 0.0
    return round(max(0.0, 5.0 - (dte) * (5.0 / 2.0)), 2)

# Risk/momentum scoring
def score_vol(vol):
    if vol is None or np.isnan(vol):
        return 2.5
    if vol <= 0:
        return 5.0
    if vol >= 0.5:
        return 0.0
    return round(max(0.0, 5.0 - (vol) * (5.0 / 0.5)), 2)

def score_max_dd(max_dd):
    if max_dd is None or np.isnan(max_dd):
        return 2.5
    dd = abs(max_dd)
    if dd <= 0.05:
        return 5.0
    if dd >= 0.8:
        return 0.0
    return round(max(0.0, 5.0 - (dd - 0.05) * (5.0 / (0.8 - 0.05))), 2)

def score_sharpe(sharpe):
    if sharpe is None or np.isnan(sharpe):
        return 2.5
    if sharpe >= 1.0:
        return 5.0
    if sharpe <= -1.0:
        return 0.0
    return round(((sharpe + 1.0) / 2.0) * 5.0, 2)

def score_ret_3m(ret):
    if ret is None or np.isnan(ret):
        return 2.5
    val = (ret + 0.5) / 1.0
    return round(min(max(val * 5.0, 0.0), 5.0), 2)

def score_rsi(rsi):
    if rsi is None or np.isnan(rsi):
        return 2.5
    return round(max(0.0, 5.0 - abs(rsi - 50) / 10.0), 2)

# High-level compute functions
def compute_value_quality_scores(fund_df):
    df = fund_df.copy()
    df["pe_score"] = df["pe"].apply(score_value_from_pe)
    df["pb_score"] = df["pb"].apply(score_pb)
    df["peg_score"] = df["peg"].apply(score_peg)
    df["div_yield_score"] = df["div_yield"].apply(score_div_yield)
    df["ev_ebitda_score"] = df["ev_ebitda"].apply(score_ev_ebitda)
    value_cols = ["pe_score", "pb_score", "peg_score", "div_yield_score", "ev_ebitda_score"]
    df["value_score"] = df[value_cols].apply(_mean_non_nan, axis=1)

    df["roe_score"] = df["roe"].apply(score_roe)
    df["rev_growth_score"] = df["rev_growth"].apply(score_rev_growth)
    df["eps_growth_score"] = df["eps_growth"].apply(score_eps_growth)
    df["fcf_yield_score"] = df["fcf_yield"].apply(score_fcf_yield)
    df["debt_to_equity_score"] = df["debt_to_equity"].apply(score_debt_to_equity)
    quality_cols = ["roe_score", "rev_growth_score", "eps_growth_score", "fcf_yield_score", "debt_to_equity_score"]
    df["quality_score"] = df[quality_cols].apply(_mean_non_nan, axis=1)

    df["value_score"] = df["value_score"].fillna(2.5)
    df["quality_score"] = df["quality_score"].fillna(2.5)
    return df

def compute_risk_momentum_scores(risk_rows_df):
    df = risk_rows_df.copy()
    df["vol_score"] = df["vol"].apply(score_vol)
    df["dd_score"] = df["max_dd"].apply(score_max_dd)
    df["sharpe_score"] = df["sharpe"].apply(score_sharpe)
    df["risk_score"] = df[["vol_score", "dd_score", "sharpe_score"]].apply(_mean_non_nan, axis=1)

    df["ret_score"] = df["ret_3m"].apply(score_ret_3m)
    df["rsi_score"] = df["rsi"].apply(score_rsi)
    df["momentum_score"] = df[["ret_score", "rsi_score"]].apply(_mean_non_nan, axis=1)

    df["risk_score"] = df["risk_score"].fillna(2.5)
    df["momentum_score"] = df["momentum_score"].fillna(2.5)
    return df
