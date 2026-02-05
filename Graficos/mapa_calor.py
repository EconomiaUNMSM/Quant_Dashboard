import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Any

# -------------------- Interactive Plotly Heatmap --------------------
def plot_panel_interactive(panel: Dict[str, Any], title: str = "Indicator Panel", height: int = 350):
    """
    Interactive Plotly heatmap for the indicator panel.
    
    Args:
        panel: Dictionary with 'trend', 'volatility', 'momentum', 'volume' keys.
        title: Chart title.
        height: Chart height in pixels.
    
    Returns:
        Plotly Figure object.
    """
    sections = ['trend', 'volatility', 'momentum']
    if panel.get('volume') is not None:
        sections.append('volume')
    
    n_rows = len(sections)
    n_cols = 5
    
    # Build intensity matrix and hover text
    z_values = []
    hover_text = []
    y_labels = []
    
    # Custom colorscale: Red -> Yellow -> Green (discrete 5 levels)
    colorscale = [
        [0.0, '#b71c1c'],   # Dark Red (Intensity 1)
        [0.25, '#ef5350'],  # Light Red (Intensity 2)
        [0.5, '#ffc107'],   # Yellow (Intensity 3)
        [0.75, '#66bb6a'],  # Light Green (Intensity 4)
        [1.0, '#1b5e20']    # Dark Green (Intensity 5)
    ]
    
    for sec in sections:
        sec_data = panel.get(sec)
        if sec_data is None:
            z_values.append([np.nan] * n_cols)
            hover_text.append(['N/A'] * n_cols)
            y_labels.append(sec.capitalize())
            continue
        
        scores = sec_data['scores']
        intens = sec_data['intensity']
        direction = sec_data.get('direction', 'neutral')
        
        keys = list(scores.index)[:5]
        if len(keys) < 5:
            keys = keys + [None] * (5 - len(keys))
        
        row_z = []
        row_hover = []
        for k in keys:
            if k is None or k not in scores.index:
                row_z.append(np.nan)
                row_hover.append('N/A')
            else:
                row_z.append(intens[k])
                row_hover.append(f"<b>{k}</b><br>Score: {scores[k]:.3f}<br>Intensity: {intens[k]}/5")
        
        z_values.append(row_z)
        hover_text.append(row_hover)
        
        if direction and direction != 'neutral':
            emoji = {'bull': '🟢', 'bear': '🔴'}.get(direction, '')
            y_labels.append(f"{sec.capitalize()} {emoji}")
        else:
            y_labels.append(sec.capitalize())
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[f"Ind #{i+1}" for i in range(n_cols)],
        y=y_labels,
        colorscale=colorscale,
        zmin=1,
        zmax=5,
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        showscale=True,
        colorbar=dict(
            title="Intensity",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['1 (Low)', '2', '3 (Mid)', '4', '5 (High)'],
            len=0.6,
            thickness=15
        )
    ))
    
    # Add text annotations on cells
    annotations = []
    for i, sec in enumerate(sections):
        sec_data = panel.get(sec)
        if sec_data is None:
            continue
        scores = sec_data['scores']
        keys = list(scores.index)[:5]
        for j, k in enumerate(keys):
            if k and k in scores.index:
                annotations.append(dict(
                    x=j,
                    y=i,
                    text=f"{k[:8]}",  # Truncate long names
                    font=dict(size=9, color='white'),
                    showarrow=False
                ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#e0e0e0')),
        template='plotly_dark',
        height=height,
        paper_bgcolor='rgba(15, 23, 36, 1)',
        plot_bgcolor='rgba(15, 23, 36, 1)',
        margin=dict(l=100, r=80, t=60, b=40),
        annotations=annotations,
        xaxis=dict(side='top')
    )
    
    return fig


# -------------------- Plotting --------------------
def plot_panel_intensity(panel: Dict[str, Any], title: str = "Quick Quant Panel", figsize=(8,4), cmap_colors=None, show=True):
    """
    Dibuja filas = secciones (trend, volatility, momentum, volume if present)
    columnas = 5 indicadores cada una (cada casilla corresponde a un indicador)
    Los colores representan la intensidad (1..5) de forma ordinal desde rojo -> green.
    """
    if cmap_colors is None:
        cmap_colors = ['#8b0000','#ff4d4d','#ffd966','#8fd19e','#0b6623']

    sections = ['trend','volatility','momentum']
    if panel.get('volume') is not None:
        sections.append('volume')

    n_rows = len(sections)
    n_cols = 5

    # construir matrix de intensities y labels
    mat = np.full((n_rows, n_cols), np.nan)
    label_mat = [['' for _ in range(n_cols)] for _ in range(n_rows)]
    dir_list = []

    for r, sec in enumerate(sections):
        sec_data = panel[sec]
        if sec_data is None:
            continue
        scores = sec_data['scores']
        intens = sec_data['intensity']
        # ensure exactly 5 indicators - if more/less, trim or pad with NaN
        keys = list(scores.index)[:5]
        # pad keys if less than 5
        if len(keys) < 5:
            keys = keys + [f"extra{i}" for i in range(5 - len(keys))]
        for c in range(5):
            k = keys[c] if c < len(scores.index) else None
            if k is None or k not in scores.index:
                mat[r, c] = np.nan
                label_mat[r][c] = ''
            else:
                mat[r, c] = intens[k]
                lab = f"{k}\n{scores[k]:.2f}"
                label_mat[r][c] = lab
        dir_list.append(sec_data.get('direction', ''))

    # plot grid of colored rectangles
    fig, ax = plt.subplots(figsize=figsize)
    # create discrete colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(np.arange(1, 7) - 0.5, cmap.N)

    # mask NaNs
    masked = np.ma.masked_invalid(mat)

    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')

    # add text labels
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(mat[i,j]):
                ax.text(j, i, label_mat[i][j], ha='center', va='center', fontsize=8, color='black')
            else:
                ax.text(j, i, '', ha='center', va='center', fontsize=8, color='black')

    # y ticks: section names and direction marker
    ytick_labels = []
    for s, d in zip(sections, dir_list):
        if d and d != 'neutral':
            ytick_labels.append(f"{s.capitalize()} ({d})")
        else:
            ytick_labels.append(s.capitalize())

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(ytick_labels)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"#{i+1}" for i in range(n_cols)])

    # legend for intensity mapping (1..5)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=cmap_colors[i], label=f"Intensity {i+1}") for i in range(5)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    ax.set_title(title)
    ax.set_xlabel("Indicators (each casilla = 1 indicator)")
    ax.grid(False)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax

if __name__ == "__main__":
    
    import yfinance as yf 
    
    df = yf.download("TSLA", period="max", interval="1d", multi_level_index=False)
    
    import sys 
    sys.path.append(r"C:\Users\LUIS\Desktop\App_trading")
    from Quant_General.Indicadores.mapa_calor import *

    # compute panel and plot
    panel = compute_panel(df)
    fig, ax = plot_panel_intensity(panel, title="Quick Quant Panel")
