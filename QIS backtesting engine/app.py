import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import metrics as mt
import base64
import os

#Page Configuration
st.set_page_config(layout="wide", page_title="Ai for Quant Research | QIS Platform")

#Helper: Image to Base64 for HTML embedding
def get_img_as_base64(file_path):
    """Encodes an image file to base64 string for HTML embedding."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load Logo
logo_filename = "Logo moderne pour AI pour QuanResearch.jpg"
logo_b64 = get_img_as_base64(logo_filename)

#CSS & Styling (Nouvelle Charte Graphique)
# Couleurs basées sur la demande : Dégradé Vert Clair -> Blanc, Accents Noirs.
st.markdown("""
    <style>
    /* MAIN BACKGROUND - Fondu vert clair à blanc */
    .stApp {
        background-image: linear-gradient(to bottom, #e8f5e9, #ffffff, #ffffff);
        color: #000000 !important; /* Force black text globally */
        font-family: 'Arial', sans-serif;
    }

    /* HEADER BOX STYLING */
    .header-container {
        display: flex;
        align-items: center;
        background: linear-gradient(to right, #e8f5e9, #ffffff);
        padding: 25px;
        border-radius: 12px;
        border-bottom: 4px solid #000000; /* Accent noir fort */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .header-logo {
        max-height: 90px;
        width: auto;
        margin-right: 25px;
        border-radius: 8px;
        border: 2px solid #000000;
    }
    .header-text h1 {
        color: #000000;
        font-weight: 900; /* Ultra Bold Black */
        font-size: 2.8rem;
        margin: 0;
        padding: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-text p {
         color: #222222;
         margin: 10px 0 0 0;
         font-weight: 700; /* Bold */
         font-size: 1.2rem;
    }

    /* SUB-HEADERS STYLING */
    .sub-header {
        color: #000000;
        border-bottom: 3px solid #000000;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 20px;
        font-weight: 900;
        font-size: 1.8rem;
    }

    /* STREAMLIT WIDGET OVERRIDES FOR HIGH VISIBILITY */
    /* Force labels to be bold black */
    .st-emotion-cache-16txtl3 label, .st-emotion-cache-10trblm, p, h1, h2, h3 {
        color: #000000 !important;
        font-weight: bold !important;
    }
    /* Dataframes with black borders */
    .stDataFrame {
        border: 2px solid #000000 !important;
    }
    /* Tabs styling */
    button[data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: bold !important;
        border-bottom: 2px solid transparent;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 4px solid #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

#HEADER SECTION (Logo + Title)
if logo_b64:
    st.markdown(f"""
        <div class="header-container">
            <img src="data:image/jpg;base64,{logo_b64}" class="header-logo" alt="Ai for Quant Research Logo">
            <div class="header-text">
                <h1>Ai for Quant Research</h1>
                <p>Quantitative Investment Strategies (QIS) - Analytics Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Fallback si le logo n'est pas trouvé
    st.markdown("""
        <div class="header-container">
            <div class="header-text">
                <h1>Ai for Quant Research</h1>
                <p>Quantitative Investment Strategies (QIS) - Analytics Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


#Data Loading (PICKLE ONLY) 
@st.cache_data
def load_data():
    data = {}
    try:
        data['lowvol_ret'] = pd.read_pickle("lowvol_returns.pkl")
        data['lowvol_pred'] = pd.read_pickle("lowvol_predictions.pkl")
        data['mom_ret'] = pd.read_pickle("momentum_returns.pkl")
        data['mom_pred'] = pd.read_pickle("momentum_predictions.pkl")
        data['deep_ret'] = pd.read_pickle("deep_returns.pkl")
        data['deep_sig'] = pd.read_pickle("deep_latest_signals.pkl")
        data['benchmark'] = pd.read_pickle("benchmark_sp500.pkl")
    except FileNotFoundError as e:
        st.error(f"System Error: Missing file {e}. Please run 'data_converter.py' first.")
        st.stop()
    return data

data = load_data()

#Helper Function for Display (Graphiques Haute Visibilité)
def display_strategy_performance(returns_df, benchmark_series, strategy_name):
    """Displays charts and metrics with HIGH VISIBILITY styling"""
    
    # 1. Equity Curve
    cum_df = (1 + returns_df).cumprod()
    df_plot = cum_df.reset_index().melt(id_vars="Date", var_name="Strategy", value_name="Index Level")
    
    fig_eq = px.line(df_plot, x="Date", y="Index Level", color="Strategy")
    
    # --- PLOTLY HIGH VISIBILITY STYLING ---
    fig_eq.update_layout(
        title=dict(text=f"{strategy_name} - Historical Performance (Log Scale)", font=dict(color='black', size=24, family="Arial Black")),
        yaxis_type="log",
        plot_bgcolor="white", # Fond blanc pur pour le graphique
        paper_bgcolor='rgba(0,0,0,0)', # Fond transparent pour le cadre
        font=dict(color='black', family="Arial"), # Police globale noire
        # Axes bien noirs et visibles
        xaxis=dict(
            showline=True, linewidth=3, linecolor='black', mirror=True,
            showgrid=True, gridcolor='#dddddd', gridwidth=1,
            tickfont=dict(weight='bold', size=12, color='black'),
            title_font=dict(weight='bold', size=14, color='black')
        ),
        yaxis=dict(
            showline=True, linewidth=3, linecolor='black', mirror=True,
            showgrid=True, gridcolor='#dddddd', gridwidth=1,
            tickfont=dict(weight='bold', size=12, color='black'),
            title_font=dict(weight='bold', size=14, color='black')
        ),
        # Légende noire et grasse
        legend=dict(
            font=dict(size=14, color="black", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black", borderwidth=2,
            title_font=dict(family="Arial Black", size=16, color="black")
        ),
        hovermode="x unified"
    )
    # Rendre les lignes du graphique plus épaisses
    fig_eq.update_traces(line=dict(width=2.5))
    
    st.plotly_chart(fig_eq, use_container_width=True)
    
    # 2. Metrics Table
    st.markdown("<h3 class='sub-header'>Risk & Performance Metrics</h3>", unsafe_allow_html=True)
    
    metrics_list = []
    for col in returns_df.columns:
        bench = benchmark_series if col != "S&P 500 Return" else None
        m = mt.compute_all_metrics(returns_df[col], bench)
        m.name = col
        metrics_list.append(m)
        
    metrics_df = pd.DataFrame(metrics_list)
    
    # Formatting
    format_dict = {
        "Annual Return": "{:.2%}", "Annual Vol": "{:.2%}", "Max DD": "{:.2%}", "VaR 95%": "{:.2%}",
        "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}", 
        "Beta": "{:.2f}", "Info Ratio": "{:.2f}", 
        "Up Capture": "{:.2f}", "Down Capture": "{:.2f}"
    }
    # Style de la table pour correspondre
    st.dataframe(
        metrics_df.style.format(format_dict, na_rep="-")
                        .set_table_styles([
                            {'selector': 'th', 'props': [('background-color', '#e8f5e9'), ('color', 'black'), ('font-weight', 'bold'), ('border', '1px solid black')]},
                            {'selector': 'td', 'props': [('color', 'black'), ('border', '1px solid black')]}
                        ]),
        use_container_width=True
    )

#Main Navigation
tabs = st.tabs(["Lab / Allocation", "Equity Strategies", "Crypto Assets", "Commodities", "Options"])

with tabs[0]: # LAB
    st.markdown("<h3 class='sub-header'>Strategy Lab: Portfolio Construction</h3>", unsafe_allow_html=True)
    st.write("Construct a custom portfolio by combining available QIS indices.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**1. Strategy Selection**")
        available_indices = {}
        def add_indices(df, prefix):
            for col in df.columns:
                if "Return" in col and "S&P" not in col and "Mkv" not in col:
                     available_indices[f"{prefix} - {col}"] = df[col]
        
        add_indices(data['mom_ret'], "Momentum")
        add_indices(data['lowvol_ret'], "LowVol")
        add_indices(data['deep_ret'], "Deep")

        selected_indices = st.multiselect("Select Indices:", list(available_indices.keys()))
        
        weights = {}
        if selected_indices:
            st.markdown("**2. Weight Allocation**")
            default_w = 1.0 / len(selected_indices)
            total_w = 0.0
            for idx in selected_indices:
                w = st.number_input(f"Weight: {idx}", min_value=0.0, max_value=1.0, value=default_w, step=0.05)
                weights[idx] = w
                total_w += w
            
            st.write(f"**Total Weight:** {total_w:.2f}")
            if abs(total_w - 1.0) > 0.01:
                st.error("Warning: Total weight must equal 1.00")

    with col2:
        if selected_indices:
            st.markdown("**3. Portfolio Backtest**")
            df_selected = pd.DataFrame({name: available_indices[name] for name in selected_indices}).dropna()
            portfolio_ret = df_selected.mul(pd.Series(weights)).sum(axis=1)
            portfolio_ret.name = "Custom Lab Portfolio"
            
            comparison_df = pd.DataFrame({
                "Custom Portfolio": portfolio_ret,
                "S&P 500 Benchmark": data['benchmark']
            }).dropna()
            
            display_strategy_performance(comparison_df, data['benchmark'], "Lab Simulation")
            
            # Correlation (High Visibility)
            st.markdown("<h3 class='sub-header'>Correlation Matrix</h3>", unsafe_allow_html=True)
            corr = df_selected.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_corr.update_layout(
                plot_bgcolor="white", paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black', family="Arial Black"),
                xaxis=dict(tickfont=dict(weight='bold', color='black'), showline=True, linewidth=2, linecolor='black'),
                yaxis=dict(tickfont=dict(weight='bold', color='black'), showline=True, linewidth=2, linecolor='black'),
                coloraxis_colorbar=dict(
                    title=dict(text="Correlation", font=dict(color="black", weight='bold')),
                    tickfont=dict(color="black", weight='bold'),
                    outlinecolor="black", outlinewidth=2
                )
            )
            st.plotly_chart(fig_corr, use_container_width=True)

with tabs[1]: # EQUITY
    st.markdown("<h3 class='sub-header'>Equity QIS Overview</h3>", unsafe_allow_html=True)
    strat_type = st.radio("Select Strategy Type:", ["Momentum", "Low Volatility", "Deep Learning / AI"], horizontal=True)
    
    if strat_type == "Momentum":
        st.markdown("#### Momentum Strategies")
        display_strategy_performance(data['mom_ret'], data['benchmark'], "Momentum")
    elif strat_type == "Low Volatility":
        st.markdown("#### Low Volatility Strategies")
        display_strategy_performance(data['lowvol_ret'], data['benchmark'], "Low Volatility")
    elif strat_type == "Deep Learning / AI":
        st.markdown("#### Deep Learning Optimized")
        display_strategy_performance(data['deep_ret'], data['benchmark'], "Deep Learning")

with tabs[2]: # CRYPTO
    st.header("Crypto Assets")
    st.info("Module under development. Data feed pending.")

with tabs[3]: # COMMODITIES
    st.header("Commodities")
    st.info("Module under development.")

with tabs[4]: # OPTIONS
    st.header("Options Strategies")
    st.info("Module under development.")