# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:07:40 2025

@author: arthu
"""

import pandas as pd
import numpy as np
import os

def convert_files():
    print(">>> Starting conversion to Pickle format...")
    
    files = {
        "lowvol_returns": ["lowvol_returns.xlsx", "lowvol_returns.xlsx - Sheet1.csv"],
        "momentum_returns": ["momentum_returns.xlsx", "momentum_returns.xlsx - Sheet1.csv"],
        "deep_returns": ["deep_returns.xlsx", "deep_returns.xlsx - Sheet1.csv"],
        "lowvol_predictions": ["lowvol_predictions.xlsx", "lowvol_predictions.xlsx - Sheet1.csv"],
        "momentum_predictions": ["momentum_predictions.xlsx", "momentum_predictions.xlsx - Sheet1.csv"],
        "deep_latest_signals": ["deep_latest_signals.xlsx", "deep_latest_signals.xlsx - Sheet1.csv"]
    }
    
    dfs = {}
    benchmark_series = None

    for key, file_list in files.items():
        loaded = False
        for f in file_list:
            if os.path.exists(f):
                print(f"   Reading {f}...")
                if f.endswith('.csv'):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
                
                # Date cleaning
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                # Save to pickle
                pkl_name = f"{key}.pkl"
                df.to_pickle(pkl_name)
                dfs[key] = df
                print(f"   [OK] Converted to {pkl_name}")
                
                # Extract S&P 500 Benchmark
                if "S&P 500 Return" in df.columns and benchmark_series is None:
                    benchmark_series = df["S&P 500 Return"].copy()
                    print("   (i) S&P 500 Benchmark extracted.")
                
                loaded = True
                break
        
        if not loaded:
            print(f"   [WARNING] Source file for {key} not found.")

    if benchmark_series is not None:
        benchmark_series.to_pickle("benchmark_sp500.pkl")
        print("   [OK] Benchmark saved as 'benchmark_sp500.pkl'")
    else:
        print("   [ERROR] Could not find 'S&P 500 Return' column to create benchmark.")

    print(">>> Conversion complete.")

if __name__ == "__main__":
    convert_files()