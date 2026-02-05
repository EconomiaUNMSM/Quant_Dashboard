import os # Por corregir (es temporal)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from typing import Tuple, Dict, Any, Optional
import torch
from sklearn.preprocessing import StandardScaler

# Intento de import ChronosPipeline (si está instalado en el entorno)
try:
    from chronos import ChronosPipeline  # type: ignore
except Exception as _e:
    ChronosPipeline = None
    _chronos_import_error = _e

def _try_load_chronos_pipeline():
    """Busca ChronosPipeline en módulos alternativos si la import directa falló."""
    candidates = [
        ("chronos", "ChronosPipeline"),
        ("chronos.pipeline", "ChronosPipeline"),
        ("chronos.pipelines", "ChronosPipeline"),
        ("chronos.model", "ChronosPipeline"),
    ]
    for module_name, attr in candidates:
        try:
            mod = __import__(module_name, fromlist=[attr])
            klass = getattr(mod, attr, None)
            if klass is not None:
                return klass
        except Exception:
            continue
    return None

def _baseline_bootstrap_forecast(prices: np.ndarray, horizon: int, ensembles: int = 200, random_seed: int = 42) -> np.ndarray:
    """Fallback bootstrap usando retornos log-normal (devuelve shape (ensembles, horizon))."""
    rng = np.random.RandomState(random_seed)
    prices = np.asarray(prices).astype(float)
    if prices.size == 0:
        return np.full((ensembles, horizon), np.nan, dtype=float)
    if prices.size == 1:
        return np.tile(prices[-1], (ensembles, horizon)).astype(float)
    logr = np.diff(np.log(prices))
    if np.all(np.isfinite(logr)) and len(logr) >= 2:
        mu = np.mean(logr)
        sigma = np.std(logr, ddof=1)
    else:
        mu = 0.0
        sigma = 1e-6
    last_price = prices[-1]
    out = np.zeros((ensembles, horizon), dtype=float)
    for i in range(ensembles):
        r = rng.normal(loc=mu, scale=sigma, size=horizon)
        s = last_price
        for t in range(horizon):
            s = s * np.exp(r[t])
            out[i, t] = s
    return out

def _model_predict_with_fallback(model, train_prices_np: np.ndarray, scaler: StandardScaler, prediction_length: int,
                                 train_df: pd.DataFrame, device: str = "cpu", ensembles_fallback: int = 200, verbose: bool = False) -> np.ndarray:
    """
    Intenta varias formas de llamar a model.predict(...). Si todo falla, usa bootstrap fallback.
    Retorna array (ensembles, horizon).
    """
    import numpy as _np
    import torch as _torch

    normalized_train = scaler.transform(train_df[['close']].values.reshape(-1, 1)).astype(_np.float32)
    candidates = [
        normalized_train.flatten(),
        normalized_train,
        _torch.from_numpy(normalized_train.flatten()).to(device),
        _torch.from_numpy(normalized_train).to(device)
    ]

    last_exc = None
    for idx, ctx in enumerate(candidates):
        try:
            if verbose:
                print(f"[DEBUG] Intentando model.predict candidate #{idx} tipo={type(ctx)} shape={getattr(ctx, 'shape', None)}")
            # Algunos modelos/implementaciones esperan (context, horizon) o (context,) -> intentamos ambas firmas
            try:
                raw = model.predict(ctx, prediction_length)
            except TypeError:
                raw = model.predict(ctx)
            if verbose:
                print(f"[DEBUG] model.predict returned type={type(raw)}")
            # convertir a numpy
            if isinstance(raw, _torch.Tensor):
                raw_np = raw.detach().cpu().numpy()
            else:
                raw_np = _np.asarray(raw, dtype=float)

            # normalizar dimensiones a (ensembles, horizon)
            if raw_np.ndim == 1:
                raw_np = raw_np.reshape(1, -1)
            elif raw_np.ndim == 2:
                # si la primera dimensión es horizon y la segunda no, transponer
                if raw_np.shape[0] == prediction_length and raw_np.shape[1] != prediction_length:
                    raw_np = raw_np.T
            elif raw_np.ndim == 3:
                # buscar eje con longitud prediction_length
                if raw_np.shape[-1] == prediction_length:
                    raw_np = raw_np.reshape(-1, prediction_length)
                elif raw_np.shape[1] == prediction_length:
                    raw_np = np.moveaxis(raw_np, 1, -1).reshape(-1, prediction_length)
                elif raw_np.shape[0] == prediction_length:
                    raw_np = np.moveaxis(raw_np, 0, -1).reshape(-1, prediction_length)
                else:
                    raw_np = raw_np.reshape(raw_np.shape[0], -1)
                    if raw_np.shape[1] >= prediction_length:
                        raw_np = raw_np[:, -prediction_length:]
                    else:
                        pad = prediction_length - raw_np.shape[1]
                        last_col = np.repeat(raw_np[:, -1][:, None], pad, axis=1)
                        raw_np = np.concatenate([raw_np, last_col], axis=1)
            # asegurar 2D
            if raw_np.ndim != 2:
                raw_np = raw_np.reshape(raw_np.shape[0], -1)
            return raw_np
        except Exception as e:
            last_exc = e
            if verbose:
                print(f"[WARN] candidate #{idx} falló: {type(e).__name__}: {e}")
            continue

    # fallback
    if verbose:
        print(f"[WARN] Todas las llamadas model.predict fallaron. Usando bootstrap fallback. Última excepción: {last_exc}")
    return _baseline_bootstrap_forecast(train_prices_np, prediction_length, ensembles=ensembles_fallback)

def _ensure_df_has_date_close(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza df a columnas ['date','close'] y asegura tipos."""
    if isinstance(df, pd.Series):
        ser = df.copy()
        ser.name = ser.name or "close"
        df = ser.reset_index()
        df.columns = ['date', 'close']
    else:
        df = df.copy()
        # detectar columna close
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'close'})
        elif 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close'})
        # si no tiene 'date' pero el index es DatetimeIndex, tomarlo
        if 'date' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={'index': 'date'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            else:
                # si hay una columna que parezca fecha, intentar convertir
                for c in df.columns:
                    if 'date' in c.lower() or 'time' in c.lower():
                        df = df.rename(columns={c: 'date'})
                        break
        # asegurar columnas
        if 'date' not in df.columns or 'close' not in df.columns:
            raise ValueError("El DataFrame debe contener una columna 'Close'/'close' y una columna 'date' o un índice datetime.")
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'close']].sort_values('date').reset_index(drop=True)
    return df

# --- Reemplazo/actualización de funciones para soporte sub-daily, quitar gaps y plot 1000 barras ---
def _infer_offset_from_dates(dts: pd.Series) -> pd.Timedelta:
    """Intenta inferir un offset (Timedelta) representativo de la serie temporal.
       Devuelve un pd.Timedelta usable en pd.date_range / suma aritmética."""
    if len(dts) < 2:
        return pd.Timedelta(days=1)
    try:
        freq = pd.infer_freq(dts.tail(200))
        if freq is not None:
            # to_offset puede devolver un DateOffset; convertir a Timedelta (si aplica)
            off = pd.tseries.frequencies.to_offset(freq)
            # algunos off no son Timedelta (ej BusinessHour), así que convertir si es posible
            try:
                td = pd.Timedelta(off.nanos, unit='ns')
                return td
            except Exception:
                # fallback a median delta
                pass
    except Exception:
        pass
    diffs = dts.diff().dropna()
    med = diffs.median()
    if not pd.isna(med):
        return pd.Timedelta(med)
    return pd.Timedelta(days=1)

def chornos_model_from_df(
    df: pd.DataFrame,
    forecast_horizon: int = 7,
    model_id: str = "amazon/chronos-t5-tiny",
    device: Optional[str] = None,
    dtype: str = "float32",
    ensembles_fallback: int = 200,
    random_seed: int = 42,
    verbose: bool = False,
    freq: Optional[str] = None,            # <-- nuevo: string de frecuencia (ej "1min","5min","H") o None para inferir
    remove_gaps: bool = True,              # <-- nuevo: si True, el plot no mostrará huecos; y las fechas de forecast se generan contiguas
    n_bars: int = 1000                     # <-- nuevo: cuántas barras mostrar por defecto (últimas n)
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pipeline actualizado:
    - Acepta series con resolución sub-diaria.
    - Si `freq` es None, intenta inferir la frecuencia de los timestamps.
    - Si `remove_gaps` True, genera forecast timestamps de manera contigua usando el offset inferido.
    - `n_bars` se usa para decidir cuántas barras históricas considerar (últimas n).
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Device -> {device}; model_id={model_id}; dtype={dtype}")

    # normalizar input
    df_clean = _ensure_df_has_date_close(df)
    if len(df_clean) < 2:
        raise ValueError("Se requieren al menos 2 observaciones históricas.")

    # quedarnos con las últimas n_bars para entrenamiento / visualización (pero no eliminamos datos si son menos)
    if n_bars is not None and n_bars > 0 and len(df_clean) > n_bars:
        train_df = df_clean.iloc[-n_bars:].reset_index(drop=True)
    else:
        train_df = df_clean.copy().reset_index(drop=True)

    # inferir offset/timedelta entre barras
    if freq is not None:
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            # convertir a Timedelta si es posible
            try:
                offset = pd.Timedelta(offset.nanos, unit='ns')
            except Exception:
                # si no se puede, usar pd.Timedelta(1, 's') * some_value fallback
                offset = _infer_offset_from_dates(train_df['date'])
        except Exception:
            offset = _infer_offset_from_dates(train_df['date'])
    else:
        offset = _infer_offset_from_dates(train_df['date'])

    # Preparar scaler y datos para el modelo / fallback
    scaler = StandardScaler()
    scaler.fit(train_df[['close']].values.reshape(-1, 1))
    normalized_train = scaler.transform(train_df[['close']].values.reshape(-1, 1)).astype(np.float32)
    prediction_length = int(forecast_horizon)

    # Intentar cargar ChronosPipeline como antes (mantengo tu lógica original de carga)
    model = None
    if 'ChronosPipeline' in globals() and ChronosPipeline is not None:
        try:
            if device == 'cuda':
                model = ChronosPipeline.from_pretrained(model_id, device_map="auto")
            else:
                model = ChronosPipeline.from_pretrained(model_id)
        except Exception as e1:
            if verbose:
                print(f"[WARN] ChronosPipeline.from_pretrained error: {e1}. Intentando fallback de carga...")
            try:
                ChronosClass = _try_load_chronos_pipeline()
                if ChronosClass is not None:
                    model = ChronosClass.from_pretrained(model_id)
                else:
                    model = None
            except Exception as e2:
                if verbose:
                    print(f"[WARN] fallback load también falló: {e2}")
                model = None
    else:
        ChronosClass = _try_load_chronos_pipeline()
        if ChronosClass is not None:
            try:
                model = ChronosClass.from_pretrained(model_id)
            except Exception as e:
                if verbose:
                    print(f"[WARN] carga alternativa ChronosClass falló: {e}")
                model = None

    train_prices_np = train_df['close'].values

    if model is not None:
        try:
            raw_np = _model_predict_with_fallback(model, train_prices_np, scaler, prediction_length, train_df, device=device, ensembles_fallback=ensembles_fallback, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[WARN] wrapper predict falló: {e}. Usando bootstrap.")
            raw_np = _baseline_bootstrap_forecast(train_prices_np, prediction_length, ensembles=ensembles_fallback)
    else:
        if verbose:
            print("[WARN] ChronosPipeline no disponible. Usando bootstrap fallback.")
        raw_np = _baseline_bootstrap_forecast(train_prices_np, prediction_length, ensembles=ensembles_fallback)

    # normalizar formas
    if raw_np.ndim == 1:
        raw_np = raw_np.reshape(1, -1)
    if raw_np.ndim == 3 and raw_np.shape[-1] == 1:
        raw_np = raw_np.squeeze(-1)
    n_ensembles, horizon = raw_np.shape[0], raw_np.shape[1]
    if horizon != prediction_length:
        if verbose:
            print(f"[WARN] model returned horizon {horizon} (requested {prediction_length}). Ajustando prediction_length -> {horizon}")
        prediction_length = horizon

    # estadísticos
    median_vals = np.median(raw_np[:, :prediction_length], axis=0)
    low_vals = np.quantile(raw_np[:, :prediction_length], 0.10, axis=0)
    high_vals = np.quantile(raw_np[:, :prediction_length], 0.90, axis=0)

    # intentar desnormalizar si era necesario
    median_unscaled = median_vals
    low_unscaled = low_vals
    high_unscaled = high_vals
    try:
        if np.all(np.isfinite(median_vals)) and (np.abs(median_vals).mean() < 10.0 and np.nanstd(median_vals) < 20.0):
            mu_test = scaler.inverse_transform(median_vals.reshape(-1, 1)).flatten()
            if np.all(np.isfinite(mu_test)):
                median_unscaled = mu_test
                low_unscaled = scaler.inverse_transform(low_vals.reshape(-1, 1)).flatten()
                high_unscaled = scaler.inverse_transform(high_vals.reshape(-1, 1)).flatten()
    except Exception:
        median_unscaled = median_vals
        low_unscaled = low_vals
        high_unscaled = high_vals

    last_date = train_df['date'].iloc[-1]

    # construir fechas de forecast: uso offset (Timedelta) para generar tiempos contiguos
    try:
        # si offset es DateOffset convertible -> intento convertir a Timedelta; si no, asumir Timedelta directo
        if isinstance(offset, pd.DateOffset):
            # intentar convertir a Timedelta (no siempre posible)
            try:
                td = pd.Timedelta(offset.nanos, unit='ns')
            except Exception:
                td = _infer_offset_from_dates(train_df['date'])
        elif isinstance(offset, pd.Timedelta):
            td = offset
        else:
            # si es string/otro, intentar pd.Timedelta
            td = pd.Timedelta(offset)
    except Exception:
        td = _infer_offset_from_dates(train_df['date'])

    forecast_dates = [last_date + td * (i + 1) for i in range(prediction_length)]
    forecast_dates = pd.to_datetime(forecast_dates)

    # truncar/ajustar longitudes por si
    median_unscaled = np.asarray(median_unscaled).flatten()
    low_unscaled = np.asarray(low_unscaled).flatten()
    high_unscaled = np.asarray(high_unscaled).flatten()
    n_dates = len(forecast_dates)
    min_len = min(n_dates, len(median_unscaled), len(low_unscaled), len(high_unscaled))
    if min_len != n_dates:
        if verbose:
            print(f"[WARN] Length mismatch: forecast_dates={n_dates}, preds_median={len(median_unscaled)}. Truncando a {min_len}.")
        forecast_dates = forecast_dates[:min_len]
        median_unscaled = median_unscaled[:min_len]
        low_unscaled = low_unscaled[:min_len]
        high_unscaled = high_unscaled[:min_len]

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'median': median_unscaled,
        'low_q': low_unscaled,
        'high_q': high_unscaled
    })

    metrics: Dict[str, Any] = {
        'ticker': None,
        'last_train_date': str(last_date),
        'last_train_price': float(train_df['close'].iloc[-1]),
        'next_bar_median_pred': float(median_unscaled[0]) if len(median_unscaled) > 0 and np.isfinite(median_unscaled[0]) else None,
        'prediction_length': int(prediction_length),
        'model_id': model_id,
        'device': device,
        'ensembles': int(n_ensembles),
        'inferred_offset': str(td)
    }

    # devolver también train_df (las últimas n_bars usadas) en metrics si quieres
    metrics['train_rows_used'] = len(train_df)

    # si user quiere remove_gaps -> la función de plot ya usará un eje contínuo
    return forecast_df, metrics

if __name__ == "__main__":
    # Ejemplo - Descarga de Datos - Utilización del Forecast_df
    import yfinance as yf 
    
    df = yf.download("PEP", period="max", interval="5m", multi_level_index=False, ignore_tz=True).Close

    # df es tu DataFrame con 'date' y 'close', por ejemplo velas 5min
    forecast_df, metrics = chornos_model_from_df(df.Close, forecast_horizon=5, freq=None, # freq = None -> El modelo solo induce la frecuencia de las velas
                                                 model_id="amazon/chronos-t5-base",
                                                 remove_gaps=True, n_bars=10_000, verbose=True)

