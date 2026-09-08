# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:01:14 2026

@author: arthu
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

def backtest_eur_gbp_usd():
    # 1. RÉCUPÉRATION DES DONNÉES
    s1, s2 = 'EURUSD=X', 'GBPUSD=X'
    print(f"Lancement du backtest pour {s1} vs {s2}...")
    
    # Décommentez pour utiliser les données réelles localement
    data = yf.download([s1, s2], start='2020-01-01', end='2024-01-01')['Close'].dropna()
    
    # Simulation (pour démonstration si yfinance est indisponible)
    # np.random.seed(42)
    # n = 1000
    # base = 1.1 + np.cumsum(np.random.normal(0, 0.005, n))
    # eurusd = base + np.random.normal(0, 0.001, n)
    # gbpusd = 1.15 * base + 0.05 + np.random.normal(0, 0.002, n)
    # data = pd.DataFrame({s1: eurusd, s2: gbpusd}, index=pd.date_range('2020-01-01', periods=n))

    # 2. PARAMÈTRES ET INITIALISATION
    window = 60
    entry_z = 1.5
    exit_z = 0.0
    fees = 0.0001 # 1 pip

    df = data.copy()
    df['hedge_ratio'] = np.nan
    df['zscore'] = np.nan

    # 3. CALCUL DU SPREAD (OLS ROULANT)
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i]
        model = sm.OLS(sub[s1], sm.add_constant(sub[s2])).fit()
        
        beta = model.params[s2]
        alpha = model.params['const']
        
        spread = df.iloc[i][s1] - (beta * df.iloc[i][s2] + alpha)
        spread_hist = sub[s1] - (beta * sub[s2] + alpha)
        z = (spread - spread_hist.mean()) / spread_hist.std()
        
        df.iloc[i, df.columns.get_loc('hedge_ratio')] = beta
        df.iloc[i, df.columns.get_loc('zscore')] = z

    # 4. LOGIQUE DE POSITIONNEMENT
    df['position'] = 0
    pos = 0
    for i in range(window, len(df)):
        z = df.iloc[i]['zscore']
        if pos == 0:
            if z > entry_z: pos = -1
            elif z < -entry_z: pos = 1
        elif (pos == 1 and z >= -exit_z) or (pos == -1 and z <= exit_z):
            pos = 0
        df.iloc[i, df.columns.get_loc('position')] = pos

    # 5. RENDEMENTS
    df['ret_s1'] = df[s1].pct_change()
    df['ret_s2'] = df[s2].pct_change()
    df['strat_ret'] = df['position'].shift(1) * (df['ret_s1'] - df['hedge_ratio'] * df['ret_s2'])
    df['strat_ret'] -= df['position'].diff().abs() * fees

    # 6. MÉTRIQUES ET GRAPHES
    cum_strat = (1 + df['strat_ret'].fillna(0)).cumprod()
    
    print(f"\nRatio de Sharpe final : {(df['strat_ret'].mean()/df['strat_ret'].std()*np.sqrt(252)):.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_strat, label='Stratégie EURUSD/GBPUSD', color='blue')
    plt.plot((1 + df['ret_s1'].fillna(0)).cumprod(), label='Benchmark EURUSD', color='gray', alpha=0.5)
    plt.title('Pairs Trading : EUR/USD vs GBP/USD')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    backtest_eur_gbp_usd()
