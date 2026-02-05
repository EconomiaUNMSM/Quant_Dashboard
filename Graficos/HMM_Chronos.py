from typing import Optional
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- Interactive Plotly Chart --------------------
def plot_candles_interactive(df_ohlc,
                              state_series,
                              forecast_df: Optional[pd.DataFrame] = None,
                              title: str = "HMM Regimes + Chronos Forecast",
                              max_bars: int = 200,
                              height: int = 600):
    """
    Interactive Plotly candlestick chart with HMM regime visualization and Chronos forecast.
    
    Args:
        df_ohlc: DataFrame with OHLC columns and DatetimeIndex.
        state_series: pd.Series with HMM labels ('bull', 'side', 'bear').
        forecast_df: Optional DataFrame with 'date', 'median', 'low_q', 'high_q'.
        title: Chart title.
        max_bars: Maximum number of bars to display.
        height: Chart height in pixels.
    
    Returns:
        Plotly Figure object.
    """
    # --- Prepare Data ---
    df = df_ohlc.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_ohlc debe tener índice DatetimeIndex.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    if max_bars is not None and max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()
    
    for c in ['Open', 'High', 'Low', 'Close']:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    if df.empty:
        raise ValueError("No hay datos válidos para plotear.")
    
    # --- Align States ---
    if not isinstance(state_series, pd.Series):
        state_series = pd.Series(state_series, index=df.index)
    
    # Ensure state_series index is timezone-naive to match df
    if hasattr(state_series.index, 'tz') and state_series.index.tz is not None:
        state_series.index = state_series.index.tz_localize(None)
    
    # Reindex states to match the trimmed df (after max_bars cut)
    # Use the exact df.index to avoid misalignment
    states = state_series.reindex(df.index, method='ffill').fillna('side')
    states = states.apply(lambda x: str(x).strip().lower() if pd.notna(x) else 'side')
    
    # --- Color mapping for regimes ---
    regime_colors = {
        'bull': '#00ff88',  # Bright green
        'bear': '#ff4d6a',  # Bright red
        'side': '#ffc107'   # Yellow
    }
    
    regime_colors_transparent = {
        'bull': 'rgba(0, 255, 136, 0.25)',
        'bear': 'rgba(255, 77, 106, 0.25)',
        'side': 'rgba(255, 193, 7, 0.15)'
    }
    
    # --- Create Figure with subplots (main chart + regime bar) ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.88, 0.12],
        subplot_titles=None
    )
    
    # --- Main Candlestick Chart ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
    ), row=1, col=1)
    
    # --- Build regime runs for background shapes ---
    runs = []
    if len(states) > 0:
        cur_label = states.iloc[0]
        run_start = df.index[0]
        for i in range(1, len(states)):
            if states.iloc[i] != cur_label:
                runs.append((run_start, df.index[i-1], cur_label))
                run_start = df.index[i]
                cur_label = states.iloc[i]
        runs.append((run_start, df.index[-1], cur_label))
    
    # --- Add regime background shapes to main chart ---
    y_min = df['Low'].min() * 0.995
    y_max = df['High'].max() * 1.005
    
    shapes = []
    for start, end, label in runs:
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=start, x1=end,
            y0=y_min, y1=y_max,
            fillcolor=regime_colors_transparent.get(label, 'rgba(128,128,128,0.1)'),
            line=dict(width=0),
            layer="below"
        ))
    
    # --- Regime Indicator Bar (bottom subplot) ---
    # Create colored bars for each regime
    for regime_name, color in regime_colors.items():
        mask = states == regime_name
        if mask.any():
            fig.add_trace(go.Bar(
                x=df.index[mask],
                y=[1] * mask.sum(),
                name=f"🔹 {regime_name.upper()}",
                marker_color=color,
                marker_line_width=0,
                showlegend=True,
                legendgroup=regime_name,
                hovertemplate=f"<b>{regime_name.upper()}</b><br>%{{x}}<extra></extra>"
            ), row=2, col=1)
    
    # --- Chronos Forecast ---
    if forecast_df is not None and not forecast_df.empty:
        fc = forecast_df.copy()
        if 'date' in fc.columns:
            fc['date'] = pd.to_datetime(fc['date']).dt.tz_localize(None)
            fc_dates = fc['date']
        else:
            fc_dates = pd.to_datetime(fc.index).tz_localize(None)
        
        # Forecast median line
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=fc['median'],
            mode='lines+markers',
            name='📈 Forecast (Median)',
            line=dict(color='#00bcd4', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond', color='#00bcd4')
        ), row=1, col=1)
        
        # Forecast confidence band
        if 'high_q' in fc.columns and 'low_q' in fc.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_dates, fc_dates[::-1]]),
                y=pd.concat([fc['high_q'], fc['low_q'][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 188, 212, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Forecast Interval',
                hoverinfo='skip',
                showlegend=False
            ), row=1, col=1)
    
    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color='#e0e0e0'),
            x=0.5,
            xanchor='center'
        ),
        template="plotly_dark",
        height=height,
        xaxis_rangeslider_visible=False,
        shapes=shapes,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=60, t=100, b=40),
        paper_bgcolor='rgba(15, 23, 36, 1)',
        plot_bgcolor='rgba(15, 23, 36, 1)',
    )
    
    # Update axes for main chart
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.05)',
        showgrid=True,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Price",
        gridcolor='rgba(255,255,255,0.05)',
        showgrid=True,
        row=1, col=1
    )
    
    # Update axes for regime bar
    fig.update_xaxes(
        title_text="Date",
        gridcolor='rgba(255,255,255,0.03)',
        showgrid=False,
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Regime",
        showticklabels=False,
        showgrid=False,
        range=[0, 1.2],
        row=2, col=1
    )
    
    # Add regime annotations on main chart
    annotations = []
    for start, end, label in runs:
        mid_time = start + (end - start) / 2
        emoji = {'bull': '🟢 BULL', 'bear': '🔴 BEAR', 'side': '🟡 SIDE'}.get(label, label.upper())
        annotations.append(dict(
            x=mid_time,
            y=y_max,
            xref='x',
            yref='y',
            text=f"<b>{emoji}</b>",
            showarrow=False,
            font=dict(size=11, color=regime_colors.get(label, '#888')),
            bgcolor='rgba(0,0,0,0.6)',
            borderpad=4,
            yshift=15
        ))
    
    fig.update_layout(annotations=annotations)
    
    return fig



def plot_candles_with_regimes_compressed(df_ohlc,
                                     state_series,
                                     forecast_df: Optional[pd.DataFrame] = None,
                                     title=None,
                                     figsize=(14,7),
                                     max_bars: int = 1000,
                                     min_run_smooth: int = 3,
                                     show_plot: bool = True,
                                     n_xticks: int = 10,
                                     forecast_median_col: str = 'median',
                                     forecast_low_col: str = 'low_q',
                                     forecast_high_col: str = 'high_q',
                                     forecast_color: str = '#1f77b4',
                                     forecast_alpha: float = 0.2):
    """
    Plot de velas OHLC + fondo por régimen, comprimido (elimina gaps de tiempo para intraday).
    - df_ohlc: DataFrame con ['Open','High','Low','Close'] y DatetimeIndex.
    - state_series: pd.Series (index datetime) o array-like labels ('bull','side','bear').
    - forecast_df: DataFrame opcional con columnas ['date' o index datetime, forecast_median_col, forecast_low_col, forecast_high_col].
                   Si se provee, se añade al final del gráfico como proyección (sin régimen).
    - max_bars: máximo número de barras a mostrar (últimas N).
    - min_run_smooth: suavizado mínimo (usa enforce_min_duration si está definida).
    - show_plot: si False devuelve (fig, ax) sin plt.show().
    - n_xticks: número aproximado de labels en eje x.
    - retorna (fig, ax).
    """

    # --- Validaciones y preparación ---
    df = df_ohlc.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_ohlc debe tener índice DatetimeIndex.")
    # normalize timezone to avoid issues
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # limitar barras (últimas max_bars)
    if max_bars is not None and max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()

    # verificar columnas OHLC
    for c in ['Open','High','Low','Close']:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Open','High','Low','Close'])
    if df.empty:
        raise ValueError("No hay datos válidos para plotear después de limpieza.")

    # -------------------
    # Alinear estados al índice de precios (merge_asof backward)
    # -------------------
    if not isinstance(state_series, pd.Series):
        try:
            state_series = pd.Series(state_series, index=df.index)
        except Exception:
            state_series = pd.Series('side', index=df.index)

    states = state_series.dropna().sort_index()
    if len(states) == 0:
        s = pd.Series('side', index=df.index)
    else:
        # si ya coincide índice -> reindex rápido
        if states.index.equals(df.index):
            s = states.reindex(df.index).fillna(method='ffill').fillna('side')
        else:
            states_df = states.reset_index()
            states_df.columns = ['date','state']
            states_df['date'] = pd.to_datetime(states_df['date']).dt.tz_localize(None)
            df_idx = pd.DataFrame({'date': df.index.values})
            merged = pd.merge_asof(df_idx, states_df, on='date', direction='backward')
            s = pd.Series(merged['state'].values, index=df.index)
            s = s.fillna(method='bfill').fillna(method='ffill').fillna('side')

    # normalizar etiquetas
    s = s.map(lambda x: str(x).strip().lower() if pd.notna(x) else 'side').fillna('side')
    allowed = {'bull','side','bear'}
    s = s.apply(lambda x: x if x in allowed else 'side')

    # suavizado de duraciones cortas si existe enforce_min_duration
    if min_run_smooth is not None and min_run_smooth > 1:
        try:
            s = enforce_min_duration(s, min_run=min_run_smooth)
        except NameError:
            # no definida: ignorar suavizado
            pass

    # -------------------
    # Construir runs (start_idx, end_idx, label) usando índice comprimido
    # -------------------
    idxs = s.index
    runs = []
    cur_label = s.iloc[0]; run_start = 0
    for i in range(1, len(idxs)):
        if s.iloc[i] != cur_label:
            run_end = i - 1
            runs.append((run_start, run_end, cur_label))
            run_start = i
            cur_label = s.iloc[i]
    runs.append((run_start, len(idxs) - 1, cur_label))

    # -------------------
    # Eje X comprimido: posiciones 0..N-1 (histórico)
    # -------------------
    N = len(df)
    x_hist = np.arange(N)   # compressed positions for history
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # cuerpo de vela ancho fijo relativo: fraction of 1 (one position)
    width = 0.6
    half_width = width / 2.0

    # -------------------
    # Procesar forecast_df (si existe)
    # -------------------
    has_forecast = False
    if forecast_df is not None:
        fc = forecast_df.copy()
        # aceptar fecha en columna 'date' o en index
        if 'date' in fc.columns:
            fc['date'] = pd.to_datetime(fc['date']).dt.tz_localize(None)
            fc_idx = fc['date'].values
        else:
            # intentar usar index
            try:
                fc_idx = pd.to_datetime(fc.index).tz_localize(None)
                fc = fc.reset_index().rename(columns={fc.columns[0]: 'date'}) if 'date' not in fc.columns else fc
            except Exception:
                fc_idx = None

        # columnas requeridas
        for col in (forecast_median_col, forecast_low_col, forecast_high_col):
            if col not in fc.columns:
                raise ValueError(f"forecast_df debe contener columna '{col}' (o ajusta forecast_*_col).")
        # limpiar NaNs
        fc = fc[[forecast_median_col, forecast_low_col, forecast_high_col] + ([ 'date' ] if 'date' in fc.columns else [])].dropna()
        if len(fc) > 0:
            has_forecast = True
            M = len(fc)
            x_fc = np.arange(N, N + M)   # posiciones comprimidas para forecast (continúa al final)
            fc_median = fc[forecast_median_col].values
            fc_low = fc[forecast_low_col].values
            fc_high = fc[forecast_high_col].values
            # for labeling ticks later, get forecast dates if available
            try:
                if 'date' in fc.columns:
                    fc_dates = pd.to_datetime(fc['date']).dt.tz_localize(None).tolist()
                else:
                    fc_dates = [None]*M
            except Exception:
                fc_dates = [None]*M
        else:
            has_forecast = False

    # -------------------
    # Plot
    # -------------------
    fig, ax = plt.subplots(figsize=figsize)

    color_map = {'bear':'#f8d7da', 'bull':'#d4edda', 'side':'#fff3cd'}
    border_map = {'bear':'#f5c6cb','bull':'#c3e6cb','side':'#ffeeba'}

    # background spans: convertir índices comprimidos a x positions
    for start_idx, end_idx, label in runs:
        start_x = start_idx - 0.5
        end_x = end_idx + 0.5
        ax.axvspan(start_x, end_x, color=color_map.get(label, 'grey'), alpha=0.5, lw=0, zorder=0)

    # dibujar velas (wick + body) usando x positions
    for xi, o, h, l, c in zip(x_hist, opens, highs, lows, closes):
        # wick
        ax.vlines(xi, l, h, linewidth=0.5, color='k', alpha=0.8, zorder=2)
        # body
        lower = min(o, c)
        height = max(abs(c - o), 1e-9)
        facecolor = '#ffffff' if c >= o else '#222222'
        edgecolor = 'k'
        rect = mpatches.Rectangle((xi - half_width, lower), width, height,
                                   facecolor=facecolor, edgecolor=edgecolor, linewidth=0.4, zorder=3)
        ax.add_patch(rect)

    # si hay forecast: dibujar línea de mediana y banda de incertidumbre
    if has_forecast:
        # trazar banda primero (debajo de la línea)
        ax.fill_between(x_fc, fc_low, fc_high, alpha=forecast_alpha, color=forecast_color, zorder=4, label='Forecast interval')
        ax.plot(x_fc, fc_median, linestyle='--', marker='o', linewidth=1.2, label='Forecast median', zorder=5, color=forecast_color)

        # separador visual entre histórico y forecast
        sep_x = N - 0.5
        ax.vlines(sep_x, np.nanmin(np.concatenate([lows, fc_low])), np.nanmax(np.concatenate([highs, fc_high])),
                  linestyle=':', linewidth=0.8, color='gray', zorder=6, alpha=0.8)
        # opcional: marcar el primer forecast timestamp con pequeño marcador en la parte superior
        ax.scatter([x_fc[0]], [fc_median[0]], marker='v', s=20, color=forecast_color, zorder=7)

    # eje X: escoger ticks legibles con labels de fecha (usar indices originales + forecast dates)
    total = N + (len(x_fc) if has_forecast else 0)
    if n_xticks is None or n_xticks <= 0:
        n_xticks = 10
    if total <= n_xticks:
        tick_pos = np.arange(total)
    else:
        step = max(1, total // n_xticks)
        tick_pos = np.arange(0, total, step)
        if tick_pos[-1] != (total - 1):
            tick_pos = np.append(tick_pos, total - 1)

    # construir labels: si pos < N -> histórico, else forecast
    tick_labels = []
    for pos in tick_pos:
        if pos < N:
            tick_labels.append(pd.to_datetime(df.index[pos]).strftime('%Y-%m-%d\n%H:%M'))
        else:
            if has_forecast:
                idx = int(pos - N)
                if idx < len(fc_dates) and fc_dates[idx] is not None:
                    tick_labels.append(pd.to_datetime(fc_dates[idx]).strftime('%Y-%m-%d\n%H:%M'))
                else:
                    tick_labels.append('')  # no hay fecha
            else:
                tick_labels.append('')

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

    ax.set_xlim(-1, total + 0.5)
    ax.set_title(title or 'Candles with Regimes (compressed) + Forecast')
    ax.set_ylabel('Price')
    ax.grid(alpha=0.2)

    # leyenda: incluir patches para regimes y forecast
    legend_patches = [mpatches.Patch(facecolor=color_map[l], edgecolor=border_map[l], label=l.capitalize()) for l in ['bull','side','bear']]
    ax.legend(handles=legend_patches + ([mpatches.Patch(facecolor=forecast_color, alpha=forecast_alpha, label='Forecast interval')] if has_forecast else []),
              loc='upper left')

    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig, ax

if __name__ == "__main__":

    # Importar Librerías Propias
    import sys 
    sys.path.append(r"C:\Users\LUIS\Desktop\App_trading")
    from Quant_General.Moldelos_Base.Chronos import *
    from Quant_General.Moldelos_Base.HMM import *

    # Extraer Datos
    import yfinance as yf 

    df = yf.download("BTC-USD", period="max", interval="1h", multi_level_index=False, ignore_tz=False)

    # Modelo HMM
    # Características - HMM
    df_feat, X = build_features(df)
    
    # Entrenamiento y Resultados
    wf = walk_forward_hmm(df_feat, X,
                          n_states=3,
                          train_window=252*2,
                          test_window=63,
                          step_size=21,
                          prob_threshold=0.6,
                          min_run=3,
                          print_probs=False,
                          save_probs_csv=None)#'probs_acn.csv')
    
    # Labels de los Estados
    labels = wf['labels'].reindex(df.index).fillna(method='ffill').fillna('side')

    # Modelo de Chornos
    # Extraer el forecast_df - Importante para la gráfica
    forecast_df, metrics = chornos_model_from_df(df.Close, forecast_horizon=5, freq=None, 
                                                 model_id="amazon/chronos-t5-base",
                                                 remove_gaps=True, n_bars=10_000, verbose=True)

    # suponiendo df_hist, labels y forecast_df ya calculados
    fig, ax = plot_candles_with_regimes_compressed(df, labels, forecast_df=forecast_df,
                                                   max_bars=200, title="Regimes (HMM) + Forecast",
                                                   n_xticks=8)

