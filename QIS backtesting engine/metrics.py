# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:06:52 2025

@author: arthu
"""

import numpy as np
import pandas as pd

def compute_all_metrics(returns_series, benchmark_series=None, risk_free_rate=0.0):
    """
    Computes professional QIS metrics: Sharpe, Sortino, Calmar, Capture Ratios, etc.
    """
    # Cleaning and Alignment
    r = returns_series.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    
    # Alignment with benchmark if provided
    if benchmark_series is not None:
        # Inner join to keep only common dates
        common_idx = r.index.intersection(benchmark_series.index)
        r = r.loc[common_idx]
        b = benchmark_series.loc[common_idx]
    else:
        b = None

    if r.empty:
        return pd.Series(dtype=float)

    #Absolute Metrics
    ann_factor = 252
    
    # 1. Annualized Return
    ann_ret = r.mean() * ann_factor
    
    # 2. Annualized Volatility
    ann_vol = r.std() * np.sqrt(ann_factor)
    
    # 3. Sharpe Ratio
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan
    
    # 4. Max Drawdown
    cum_ret = (1 + r).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    # 5. Calmar Ratio (Ann Return / abs(MaxDD))
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    # 6. Sortino Ratio
    # Downside Deviation: std dev of negative returns only
    neg_ret = r[r < 0]
    downside_std = np.sqrt((neg_ret**2).mean()) * np.sqrt(ann_factor)
    sortino = (ann_ret - risk_free_rate) / downside_std if downside_std != 0 else np.nan

    # 7. VaR 95% (Historical)
    var_95 = np.percentile(r, 5)

    metrics = {
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD": max_dd,
        "VaR 95%": var_95
    }

    # Relative Metrics (if Benchmark is present) 
    if b is not None and not b.empty:
        # 8. Beta
        cov = np.cov(r, b)[0, 1]
        var_b = np.var(b)
        beta = cov / var_b if var_b != 0 else np.nan
        
        # 9. Information Ratio
        active_ret = r - b
        tracking_error = active_ret.std() * np.sqrt(ann_factor)
        ir = (active_ret.mean() * ann_factor) / tracking_error if tracking_error != 0 else np.nan
        
        # 10. Up/Down Capture Ratios
        # Up Capture
        up_idx = b > 0
        if up_idx.sum() > 0:
            up_r_geo = (1 + r[up_idx]).prod()**(1/up_idx.sum()) - 1
            up_b_geo = (1 + b[up_idx]).prod()**(1/up_idx.sum()) - 1
            up_capture = up_r_geo / up_b_geo if up_b_geo != 0 else np.nan
        else:
            up_capture = np.nan
            
        # Down Capture
        down_idx = b < 0
        if down_idx.sum() > 0:
            down_r_geo = (1 + r[down_idx]).prod()**(1/down_idx.sum()) - 1
            down_b_geo = (1 + b[down_idx]).prod()**(1/down_idx.sum()) - 1
            down_capture = down_r_geo / down_b_geo if down_b_geo != 0 else np.nan
        else:
            down_capture = np.nan
            
        metrics.update({
            "Beta": beta,
            "Info Ratio": ir,
            "Up Capture": up_capture,
            "Down Capture": down_capture
        })
    else:
        metrics.update({
            "Beta": np.nan, "Info Ratio": np.nan, "Up Capture": np.nan, "Down Capture": np.nan
        })

    return pd.Series(metrics)