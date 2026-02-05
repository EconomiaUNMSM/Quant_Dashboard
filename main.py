# Importar Librerías Propias
import os
import sys 

# --- Import dynamic path resolution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Modelos - Uitlidades
from Quant_General.Moldelos_Base.Chronos import *
from Quant_General.Moldelos_Base.HMM import *
from Quant_General.Moldelos_Base.Finbert import *
from Quant_General.Indicadores.mapa_calor import *

# Gráficos
from Quant_General.Graficos.HMM_Chronos import *
from Quant_General.Graficos.mapa_calor import *

# Datos
from Quant_General.Fuente_Datos import * # Por agregar - manipular

# --------------------------- Datos Extraidos ---------------------------#
import yfinance as yf 
# Ticker
ticker = "BTC-USD"

df = yf.download(ticker, period="max", interval="1h", multi_level_index=False, ignore_tz=False)

# --------------------------- Modelo HMM ---------------------------#
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

# --------------------------- Modelo Chronos ---------------------------#
# Extraer el forecast_df - Importante para la gráfica
forecast_df, metrics = chornos_model_from_df(df.Close, forecast_horizon=5, freq=None, 
                                                model_id="amazon/chronos-t5-base",
                                                remove_gaps=True, n_bars=10_000, verbose=True)

# --------------------------- Modelo Finbert ---------------------------#
# Tabla de Analisis de Sentimiento - Consulta Editable (10 últimas noticias relaciondas)
df = search_news_sentiment(query=ticker)

# --------------------------- Graficos ---------------------------#
# suponiendo df_hist, labels y forecast_df ya calculados
fig, ax = plot_candles_with_regimes_compressed(df, labels, forecast_df=forecast_df,
                                                max_bars=100, title="Regimes (HMM) + Forecast",
                                                n_xticks=8)
# compute panel and plot
panel = compute_panel(df)
fig, ax = plot_panel_intensity(panel, title="Quick Quant Panel")
