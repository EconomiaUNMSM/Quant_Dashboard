import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Premium Glassmorphism CSS
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0f1724 50%, #071026 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Title */
    .main-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }
    
    .sub-title {
        color: #6b7b8c;
        font-size: 14px;
        font-weight: 400;
        margin-bottom: 20px;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .glass-card-accent {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        flex: 1;
        text-align: center;
    }
    
    .kpi-label {
        font-size: 11px;
        color: #6b7b8c;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 22px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        color: #e0e0e0;
    }
    
    .kpi-value.positive { color: #00ff88; }
    .kpi-value.negative { color: #ff4d6a; }
    .kpi-value.bull { color: #00ff88; }
    .kpi-value.bear { color: #ff4d6a; }
    .kpi-value.side { color: #ffc107; }
    
    /* Section Headers */
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-header::before {
        content: '';
        width: 4px;
        height: 16px;
        background: linear-gradient(180deg, #00d4ff, #00ff88);
        border-radius: 2px;
    }
    
    /* Finbert Table */
    .finbert-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    
    .finbert-table th {
        background: rgba(0, 212, 255, 0.08);
        color: #00d4ff;
        padding: 10px 8px;
        text-align: left;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.5px;
    }
    
    .finbert-table td {
        padding: 10px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #c0c0c0;
    }
    
    .finbert-table tr:hover {
        background: rgba(0, 212, 255, 0.05);
    }
    
    .finbert-container {
        max-height: 350px;
        overflow-y: auto;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 36, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox select {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(0, 0, 0, 0.2);
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #6b7b8c;
        font-weight: 500;
        padding: 8px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 255, 136, 0.1));
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(0, 255, 136, 0.2));
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Dynamic Path Resolution
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

IMPORT_ERR = None
try:
    from Quant_General.Moldelos_Base.Chronos import chornos_model_from_df
    from Quant_General.Moldelos_Base.HMM import build_features, walk_forward_hmm
    from Quant_General.Moldelos_Base.Finbert import search_news_sentiment
    from Quant_General.Indicadores.mapa_calor import compute_panel
    from Quant_General.Graficos.HMM_Chronos import plot_candles_interactive, plot_candles_with_regimes_compressed
    from Quant_General.Graficos.mapa_calor import plot_panel_interactive, plot_panel_intensity
except Exception as e:
    IMPORT_ERR = e

# -------------------------
# Cached Data Functions
# -------------------------
@st.cache_data
def fetch_data(ticker: str, interval: str):
    import yfinance as yf
    df = yf.download(ticker, period='max', interval=interval, multi_level_index=False, ignore_tz=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    # Normalize index to timezone-naive
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            df[c] = np.nan
    return df

@st.cache_data
def run_hmm_cached(ticker: str, interval: str):
    df = fetch_data(ticker, interval)
    # Ensure df index is timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df_feat, X = build_features(df)
    wf = walk_forward_hmm(df_feat, X, n_states=3, train_window=252*2, test_window=63,
                          step_size=21, prob_threshold=0.6, min_run=3, print_probs=False, save_probs_csv=None)
    
    if isinstance(wf, dict) and 'labels' in wf:
        labels = wf['labels'].copy()
        # Normalize labels index to timezone-naive
        if hasattr(labels.index, 'tz') and labels.index.tz is not None:
            labels.index = labels.index.tz_localize(None)
        else:
            labels.index = pd.to_datetime(labels.index)
            if labels.index.tz is not None:
                labels.index = labels.index.tz_localize(None)
        # Now reindex safely
        labels = labels.reindex(df.index, method='ffill').fillna('side')
    else:
        labels = pd.Series(index=df.index, data='side')
    
    return wf, labels, df

@st.cache_data
def run_chronos_cached(ticker: str, interval: str, forecast_horizon: int):
    df = fetch_data(ticker, interval)
    forecast_df, metrics = chornos_model_from_df(df.Close, forecast_horizon=forecast_horizon, freq=None,
                                                 model_id="amazon/chronos-t5-base",
                                                 remove_gaps=True, n_bars=10_000, verbose=False)
    return forecast_df, metrics

@st.cache_data
def run_finbert_cached(ticker: str):
    df_news = search_news_sentiment(query=ticker)
    if df_news is None:
        return pd.DataFrame(columns=['Fecha', 'Hora', 'Titular', 'pos_%', 'neg_%', 'Gap'])
    return df_news.head(10)

@st.cache_data
def compute_panel_cached(ticker: str, interval: str):
    df = fetch_data(ticker, interval)
    return compute_panel(df)

# -------------------------
# Main App
# -------------------------
def main():
    # Header
    st.markdown('<div class="main-title">📊 Quant Terminal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">HMM Regime Detection • Chronos Forecasting • FinBERT Sentiment Analysis</div>', unsafe_allow_html=True)
    
    if IMPORT_ERR:
        st.error(f"Import Error: {IMPORT_ERR}")
        st.stop()
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        ticker = st.text_input("Ticker Symbol", value="BTC-USD", placeholder="e.g. AAPL, BTC-USD")
        interval = st.selectbox("Interval", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk'], index=5)
        forecast_horizon = st.slider("Forecast Horizon (bars)", 1, 30, 5)
        max_bars = st.slider("Chart Bars", 50, 500, 150)
        
        st.markdown("---")
        st.markdown("### 🔄 Actions")
        refresh_all = st.button("🚀 Run Full Analysis")
        refresh_news = st.button("📰 Refresh Sentiment")
    
    if not ticker:
        st.info("👆 Enter a ticker symbol in the sidebar to begin.")
        st.stop()
    
    # Data Loading with Status
    with st.status("Loading data and models...", expanded=True) as status:
        st.write("📥 Fetching market data...")
        try:
            df = fetch_data(ticker, interval)
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            st.stop()
        
        st.write("🧠 Running HMM analysis...")
        try:
            wf, labels, df_hmm = run_hmm_cached(ticker, interval)
            current_regime = str(labels.iloc[-1]).strip().lower() if len(labels) > 0 else 'side'
            # Debug: show label distribution
            st.write(f"📊 HMM Labels: {labels.value_counts().to_dict()}")
        except Exception as e:
            st.warning(f"HMM error: {e}")
            labels = pd.Series(index=df.index, data='side')
            current_regime = 'side'
        
        st.write("🔮 Generating Chronos forecast...")
        try:
            forecast_df, metrics = run_chronos_cached(ticker, interval, forecast_horizon)
        except Exception as e:
            st.warning(f"Chronos error: {e}")
            forecast_df = pd.DataFrame()
        
        st.write("📊 Computing indicators...")
        try:
            panel = compute_panel_cached(ticker, interval)
        except Exception as e:
            st.warning(f"Panel error: {e}")
            panel = {}
        
        st.write("📰 Fetching sentiment...")
        try:
            fin_df = run_finbert_cached(ticker)
            avg_gap = fin_df['Gap'].mean() if 'Gap' in fin_df.columns and not fin_df.empty else 0
        except Exception:
            fin_df = pd.DataFrame()
            avg_gap = 0
        
        status.update(label="✅ Analysis complete!", state="complete")
    
    # KPI Row
    last_price = df['Close'].iloc[-1] if not df.empty else 0
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else last_price
    pct_change = ((last_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
    
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Last Price</div>
            <div class="kpi-value">${last_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[1]:
        pct_class = "positive" if pct_change >= 0 else "negative"
        pct_sign = "+" if pct_change >= 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Change</div>
            <div class="kpi-value {pct_class}">{pct_sign}{pct_change:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[2]:
        regime_emoji = {'bull': '🟢', 'bear': '🔴', 'side': '🟡'}.get(current_regime, '⚪')
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">HMM Regime</div>
            <div class="kpi-value {current_regime}">{regime_emoji} {current_regime.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[3]:
        sent_class = "positive" if avg_gap > 0 else "negative" if avg_gap < 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Sentiment Gap</div>
            <div class="kpi-value {sent_class}">{avg_gap:+.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Market Vision", "🔬 Deep Analysis", "📄 Raw Data"])
    
    with tab1:
        st.markdown('<div class="section-header">Interactive Price Chart with HMM Regimes</div>', unsafe_allow_html=True)
        try:
            fig = plot_candles_interactive(df, labels, forecast_df, title=f"{ticker} — {interval}", max_bars=max_bars, height=550)
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Chart error: {e}")
            # Fallback to matplotlib
            try:
                fig_mpl, _ = plot_candles_with_regimes_compressed(df, labels, forecast_df, max_bars=max_bars, title=ticker, show_plot=False)
                st.pyplot(fig_mpl)
            except:
                st.warning("Could not render chart.")
    
    with tab2:
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.markdown('<div class="section-header">Indicator Heatmap</div>', unsafe_allow_html=True)
            try:
                fig_panel = plot_panel_interactive(panel, title="Technical Indicators", height=320)
                st.plotly_chart(fig_panel)
            except Exception as e:
                st.warning(f"Heatmap error: {e}")
                st.write(panel)
        
        with col_right:
            st.markdown('<div class="section-header">FinBERT Sentiment</div>', unsafe_allow_html=True)
            if fin_df is not None and not fin_df.empty:
                html = fin_df.to_html(index=False, classes='finbert-table', escape=True)
                st.markdown(f'<div class="finbert-container">{html}</div>', unsafe_allow_html=True)
            else:
                st.info("No news data available.")
    
    with tab3:
        st.markdown('<div class="section-header">Forecast Data</div>', unsafe_allow_html=True)
        if not forecast_df.empty:
            st.dataframe(forecast_df)
        else:
            st.info("No forecast data.")
        
        st.markdown('<div class="section-header">OHLCV Data (Last 50)</div>', unsafe_allow_html=True)
        st.dataframe(df.tail(50))

if __name__ == '__main__':
    main()
