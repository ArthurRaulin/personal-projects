# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:57:58 2026

@author: arthu
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

def backtest_eurjpy_gbpjpy():
    # 1. TÉLÉCHARGEMENT DES DONNÉES
    s1, s2 = 'EURJPY=X', 'GBPJPY=X'
    print(f"Téléchargement de {s1} et {s2}...")
    data = yf.download([s1, s2], start='2020-01-01', end='2024-01-01')['Close'].dropna()
    
    # Paramètres de la stratégie (basés sur vos notebooks)
    window = 60      
    entry_z = 1.5    
    exit_z = 0.0     
    fees = 0.0001    # 1 pip de frais

    df = data.copy()
    df['hedge_ratio'] = np.nan
    df['zscore'] = np.nan

    # 2. CALCUL DU SPREAD DYNAMIQUE (OLS ROULANT)
    print("Calcul des signaux (Z-Score)...")
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i]
        # On régresse EURJPY sur GBPJPY
        y = sub[s1]
        x = sm.add_constant(sub[s2])
        model = sm.OLS(y, x).fit()
        
        beta = model.params[s2]
        alpha = model.params['const']
        
        # Spread actuel et normalisation
        current_spread = df.iloc[i][s1] - (beta * df.iloc[i][s2] + alpha)
        spread_hist = sub[s1] - (beta * sub[s2] + alpha)
        z = (current_spread - spread_hist.mean()) / spread_hist.std()
        
        df.iloc[i, df.columns.get_loc('hedge_ratio')] = beta
        df.iloc[i, df.columns.get_loc('zscore')] = z

    # 3. GÉNÉRATION DES POSITIONS
    df['position'] = 0
    curr_pos = 0
    for i in range(window, len(df)):
        z = df.iloc[i]['zscore']
        if curr_pos == 0:
            if z > entry_z: curr_pos = -1  # Vendre le spread
            elif z < -entry_z: curr_pos = 1 # Acheter le spread
        elif (curr_pos == 1 and z >= -exit_z) or (curr_pos == -1 and z <= exit_z):
            curr_pos = 0
        df.iloc[i, df.columns.get_loc('position')] = curr_pos

    # 4. CALCUL DES RENDEMENTS
    df['ret_s1'] = df[s1].pct_change()
    df['ret_s2'] = df[s2].pct_change()
    
    # Performance = Position * (Variation S1 - Beta * Variation S2)
    df['strat_ret'] = df['position'].shift(1) * (df['ret_s1'] - df['hedge_ratio'] * df['ret_s2'])
    
    # Déduction des frais lors de chaque changement de position
    df['trades'] = df['position'].diff().fillna(0).abs()
    df['strat_ret'] = df['strat_ret'] - (df['trades'] * fees)

    # 5. MÉTRIQUES ET AFFICHAGE
    rets = df['strat_ret'].fillna(0)
    cum_strat = (1 + rets).cumprod()
    cum_bench = (1 + df['ret_s1'].fillna(0)).cumprod() # Benchmark = Hold EURJPY

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    max_dd = (cum_strat / cum_strat.cummax() - 1).min()

    print("\n" + "="*30)
    print(f"RÉSULTATS : {s1} / {s2}")
    print(f"Rendement Annuel : {ann_ret*100:.2f}%")
    print(f"Volatilité Annuelle : {ann_vol:.2f}%")
    print(f"Ratio de Sharpe : {sharpe:.2f}")
    print(f"Max Drawdown : {max_dd*100:.2f}%")
    print("="*30)

    # Graphique
    plt.figure(figsize=(12, 6))
    plt.plot(cum_strat, label='Stratégie Pairs Trading (Market Neutral)', color='#1f77b4', lw=2)
    plt.plot(cum_bench, label=f'Benchmark (Hold {s1})', color='#d62728', lw=1, ls='--')
    plt.title(f'Backtest {s1} vs {s2} (Corrélation)', fontsize=14)
    plt.ylabel('Valeur Cumulative')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    backtest_eurjpy_gbpjpy()
