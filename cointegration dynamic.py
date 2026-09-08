import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

def run_forex_strategy_spyder():
    print("Étape 1 : Téléchargement des données...")
    # Téléchargement via yfinance
    tickers = ['AUDUSD=X', 'NZDUSD=X']
    data = yf.download(tickers, start='2021-01-01', end='2024-01-01')['Close']
    
    if data.empty:
        print("Erreur : Impossible de récupérer les données. Vérifiez votre connexion.")
        return

    data.columns = ['AUDUSD', 'NZDUSD']
    data = data.dropna()
    print(f"Données récupérées : {len(data)} jours.")

    # Paramètres
    window = 60
    entry_z = 1.5
    exit_z = 0.0
    fees = 0.0001 

    results = data.copy()
    results['hedge_ratio'] = np.nan
    results['zscore'] = np.nan

    print("Étape 2 : Calcul de la cointégration dynamique...")
    for i in range(window, len(data)):
        sub = data.iloc[i-window:i]
        y = sub['AUDUSD']
        x = sm.add_constant(sub['NZDUSD'])
        model = sm.OLS(y, x).fit()
        
        beta = model.params['NZDUSD']
        alpha = model.params['const']
        
        current_spread = data.iloc[i]['AUDUSD'] - (beta * data.iloc[i]['NZDUSD'] + alpha)
        spread_hist = sub['AUDUSD'] - (beta * sub['NZDUSD'] + alpha)
        z = (current_spread - spread_hist.mean()) / spread_hist.std()
        
        results.iloc[i, results.columns.get_loc('hedge_ratio')] = beta
        results.iloc[i, results.columns.get_loc('zscore')] = z

    # Signaux
    results['position'] = 0
    pos = 0
    for i in range(window, len(results)):
        z = results.iloc[i]['zscore']
        if pos == 0:
            if z > entry_z: pos = -1
            elif z < -entry_z: pos = 1
        elif (pos == 1 and z >= -exit_z) or (pos == -1 and z <= exit_z):
            pos = 0
        results.iloc[i, results.columns.get_loc('position')] = pos

    # Calcul Performance
    results['ret_aud'] = results['AUDUSD'].pct_change()
    results['ret_nzd'] = results['NZDUSD'].pct_change()
    results['strat_ret'] = results['position'].shift(1) * (results['ret_aud'] - results['hedge_ratio'] * results['ret_nzd'])
    results['strat_ret'] -= results['position'].diff().abs() * fees
    
    results['cum_strat'] = (1 + results['strat_ret'].fillna(0)).cumprod()
    results['cum_bench'] = (1 + results['ret_aud'].fillna(0)).cumprod()

    # Affichage du graphique
    print("Étape 3 : Génération du graphique...")
    plt.figure(figsize=(10, 6))
    plt.plot(results['cum_strat'], label='Stratégie Pairs Trading', color='blue')
    plt.plot(results['cum_bench'], label='Benchmark (AUDUSD)', color='red', linestyle='--')
    plt.title('Performance AUD/NZD - Cointégration Dynamique')
    plt.legend()
    plt.grid(True)
    plt.show() # Crucial pour Spyder

    # Affichage des métriques dans la console
    print("\n--- RÉSUMÉ DES MÉTRIQUES ---")
    ann_ret = (results['strat_ret'].mean() * 252) * 100
    ann_vol = (results['strat_ret'].std() * np.sqrt(252)) * 100
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    print(f"Rendement Annuel : {ann_ret:.2f}%")
    print(f"Volatilité Annuelle : {ann_vol:.2f}%")
    print(f"Ratio de Sharpe : {sharpe:.2f}")

if __name__ == "__main__":
    run_forex_strategy_spyder()