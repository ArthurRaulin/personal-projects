# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:53:36 2026

@author: arthu
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations

# 1. Configuration et Téléchargement
tickers = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X', 
           'USDCAD=X', 'USDCHF=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X']

print("Téléchargement des données pour la sélection...")
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close'].dropna()

def find_cointegrated_pairs(df):
    n = df.shape[1]
    pairs = []
    keys = df.columns
    for i, j in combinations(range(n), 2):
        s1 = df[keys[i]]
        s2 = df[keys[j]]
        # Régression pour obtenir les résidus
        result = sm.OLS(s1, sm.add_constant(s2)).fit()
        # Test ADF sur les résidus
        p_val = adfuller(result.resid)[1]
        if p_val < 0.05: # Seuil de confiance 95%
            pairs.append({'Pair': f"{keys[i]} / {keys[j]}", 'P-Value': p_val})
    
    return pd.DataFrame(pairs).sort_values(by='P-Value')

print("\n--- MEILLEURES PAIRES PAR COINTÉGRATION ---")
coint_results = find_cointegrated_pairs(data)
print(coint_results.head(10))

###########################################################################
def find_correlated_pairs(df, threshold=0.85):
    corr_matrix = df.corr()
    pairs = []
    keys = df.columns
    for i, j in combinations(range(len(keys)), 2):
        c = corr_matrix.iloc[i, j]
        if abs(c) > threshold:
            pairs.append({'Pair': f"{keys[i]} / {keys[j]}", 'Correlation': c})
    
    return pd.DataFrame(pairs).sort_values(by='Correlation', ascending=False)

print("\n--- MEILLEURES PAIRES PAR CORRÉLATION ---")
corr_results = find_correlated_pairs(data)
print(corr_results.head(10))

###########################################################################
import matplotlib.pyplot as plt

# --- CONFIGURATION DU BACKTEST ---
S1 = 'EURUSD=X' 
S2 = 'GBPUSD=X'
# ---------------------------------

def backtest_engine(df, s1, s2, window=60, entry_z=1.5, exit_z=0.0):
    df_pair = df[[s1, s2]].copy()
    df_pair['hedge_ratio'] = np.nan
    df_pair['zscore'] = np.nan
    
    # OLS Roulant pour Beta et Z-Score
    for i in range(window, len(df_pair)):
        sub = df_pair.iloc[i-window:i]
        model = sm.OLS(sub[s1], sm.add_constant(sub[s2])).fit()
        beta = model.params[s2]
        alpha = model.params['const']
        
        spread = df_pair.iloc[i][s1] - (beta * df_pair.iloc[i][s2] + alpha)
        spread_hist = sub[s1] - (beta * sub[s2] + alpha)
        z = (spread - spread_hist.mean()) / spread_hist.std()
        
        df_pair.iloc[i, df_pair.columns.get_loc('hedge_ratio')] = beta
        df_pair.iloc[i, df_pair.columns.get_loc('zscore')] = z

    # Signaux
    df_pair['position'] = 0
    pos = 0
    for i in range(window, len(df_pair)):
        z = df_pair.iloc[i]['zscore']
        if pos == 0:
            if z > entry_z: pos = -1
            elif z < -entry_z: pos = 1
        elif (pos == 1 and z >= -exit_z) or (pos == -1 and z <= exit_z):
            pos = 0
        df_pair.iloc[i, df_pair.columns.get_loc('position')] = pos

    # Rendements
    df_pair['ret_s1'] = df_pair[s1].pct_change()
    df_pair['ret_s2'] = df_pair[s2].pct_change()
    df_pair['strat_ret'] = df_pair['position'].shift(1) * (df_pair['ret_s1'] - df_pair['hedge_ratio'] * df_pair['ret_s2'])
    
    # Métriques
    rets = df_pair['strat_ret'].fillna(0)
    ann_ret = (rets.mean() * 252)
    ann_vol = (rets.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    cum_ret = (1 + rets).cumprod()
    dd = (cum_ret / cum_ret.cummax()) - 1
    
    return df_pair, {"Return": ann_ret, "Vol": ann_vol, "Sharpe": sharpe, "MaxDD": dd.min()}

# Exécution
final_df, metrics = backtest_engine(data, S1, S2)

# Affichage
print(f"\n--- RÉSULTATS BACKTEST {S1}/{S2} ---")
for k, v in metrics.items(): print(f"{k}: {v:.4f}")

plt.figure(figsize=(12, 6))
plt.plot((1 + final_df['strat_ret'].fillna(0)).cumprod(), label='Stratégie Pairs Trading', color='blue')
plt.plot((1 + final_df[S1].pct_change().fillna(0)).cumprod(), label=f'Benchmark ({S1})', color='gray', alpha=0.5)
plt.title(f'Performance : {S1} vs {S2}')
plt.legend()
plt.grid(True)
plt.show()
