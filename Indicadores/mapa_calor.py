import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any

# -------------------- Utilities de indicadores --------------------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def pct_change(series: pd.Series, periods=1):
    return series.pct_change(periods=periods)

def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, length=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def adx(df: pd.DataFrame, length=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_smooth = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_dm_smooth = plus_dm.ewm(alpha=1/length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-12))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-12))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx, plus_di, minus_di

def bollinger_bandwidth(series: pd.Series, length=20, ndev=2):
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = ma + ndev * std
    lower = ma - ndev * std
    bw = (upper - lower) / (ma + 1e-12)
    return bw

def parkinson_volatility(df: pd.DataFrame, length=20):
    # Parkinson estimator over rolling window
    hl = np.log(df['High'] / df['Low']) ** 2
    pv = (1.0 / (4.0 * np.log(2))) * hl.rolling(length).sum() / length
    pv = np.sqrt(pv)  # as volatility
    return pv

def stochastic_k(df: pd.DataFrame, k_period=14, d_period=3):
    low_k = df['Low'].rolling(k_period).min()
    high_k = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_k) / (high_k - low_k + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d

def roc(series: pd.Series, lookback=12):
    return series.pct_change(periods=lookback)

def obv(df: pd.DataFrame):
    vol = df.get('Volume', pd.Series(0, index=df.index))
    direction = np.sign(df['Close'].diff().fillna(0))
    return (direction * vol).cumsum()

def money_flow_index(df: pd.DataFrame, length=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    mf = typical_price * df['Volume']
    positive_mf = mf.where(typical_price > typical_price.shift(), 0.0)
    negative_mf = mf.where(typical_price < typical_price.shift(), 0.0)
    positive_mf_sum = positive_mf.rolling(length).sum()
    negative_mf_sum = negative_mf.rolling(length).sum().abs() + 1e-12
    mfi = 100 - 100 / (1 + (positive_mf_sum / negative_mf_sum))
    return mfi

def vwap(df: pd.DataFrame, length=None):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    if length is None:
        # cumulative vwap
        cum_pv = (tp * df.get('Volume', pd.Series(0, index=df.index))).cumsum()
        cum_v = df.get('Volume', pd.Series(0, index=df.index)).cumsum() + 1e-12
        return cum_pv / cum_v
    else:
        pv = (tp * df['Volume']).rolling(length).sum()
        v = df['Volume'].rolling(length).sum() + 1e-12
        return pv / v

def atr_ratio(df: pd.DataFrame, length=14):
    a = atr(df, length=length)
    return a / (df['Close'].rolling(length).mean() + 1e-12)

# percentile rank of last value relative to historical series
def percentile_rank(series: pd.Series, value: float):
    # fraction of values <= value
    ser = series.dropna()
    if len(ser) == 0:
        return 0.5
    return float((ser <= value).sum()) / float(len(ser))

# map score [0,1] to intensity 1..5
def score_to_intensity(score: float, n_bins=5):
    score_clipped = float(np.clip(score, 0.0, 1.0))
    # bins [0,0.2), [0.2,0.4), ... map to 1..5
    idx = int(np.floor(score_clipped * n_bins))
    if idx == n_bins:
        idx = n_bins - 1
    return idx + 1  # 1..5

# -------------------- Panel computation --------------------
def compute_panel(df: pd.DataFrame,
                  lookback_hist:int = 252,
                  trend_lookback = 20,
                  vol_lookback = 20,
                  mom_lookback = 20,
                  volma_lookback = 20):
    """
    Devuelve dict con DataFrames de 'scores' (0..1), 'intensity' (1..5) por sección y el 'direction' para tendencia/momentum.
    """
    out = {}
    close = df['Close']

    # --- Trend indicators (5) ---
    # 1. EMA slope (EMA20 slope over last 5 bars)
    ema20 = ema(close, span=20)
    ema20_slope = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-12)
    ema20_slope_hist = ema20_slope.dropna()

    # 2. ADX (14)
    adx_series, plus_di, minus_di = adx(df, length=14)
    adx_hist = adx_series.dropna()

    # 3. Price above EMA50 (fraction over last trend_lookback)
    ema50 = ema(close, span=50)
    price_above_ema50 = (close > ema50).rolling(window=trend_lookback, min_periods=1).mean()

    # 4. MACD histogram magnitude (mean abs hist over lookback)
    macd_line, signal_line, macd_hist = macd(close, fast=12, slow=26, signal=9)
    macd_hist_abs = macd_hist.abs().rolling(window=trend_lookback, min_periods=1).mean()

    # 5. Linear regression slope (normalized)
    def linreg_slope(s: pd.Series, window=20):
        idx = np.arange(window)
        res = pd.Series(np.nan, index=s.index)
        for i in range(window - 1, len(s)):
            y = s.iloc[i-window+1:i+1].values
            x = idx
            # slope via polyfit
            slope = np.polyfit(x, y, 1)[0]
            res.iloc[i] = slope
        return res
    lr_slope = linreg_slope(close, window=trend_lookback)

    # assemble history series for percentiles (we compute percentiles using last `lookback_hist` samples)
    hist_slice_start = max(0, len(df) - lookback_hist)
    idx_hist = df.index[hist_slice_start:]

    # helper to calc score by percentile relative to historical series
    def score_from_series(series: pd.Series):
        ser_hist = series.loc[idx_hist].dropna() if len(series.loc[idx_hist].dropna())>0 else series.dropna()
        if len(ser_hist) == 0:
            return 0.5
        return percentile_rank(ser_hist, series.iloc[-1])

    trend_scores = {}
    # note: for indicators where higher is "more trending", we keep as is.
    trend_scores['EMA20_slope'] = score_from_series(ema20_slope)
    trend_scores['ADX14'] = score_from_series(adx_series)
    trend_scores['Price>EMA50_frac'] = score_from_series(price_above_ema50)
    trend_scores['MACD_hist_mag'] = score_from_series(macd_hist_abs)
    trend_scores['LR_slope'] = score_from_series(lr_slope)

    trend_scores_df = pd.Series(trend_scores)
    trend_intensity = trend_scores_df.apply(score_to_intensity)

    # direction: bullish vs bearish signature from last close vs ema20 and macd sign
    direction_trend = 'neutral'
    try:
        last_close = close.iloc[-1]
        if last_close > ema20.iloc[-1] and macd_hist.iloc[-1] > 0:
            direction_trend = 'bull'
        elif last_close < ema20.iloc[-1] and macd_hist.iloc[-1] < 0:
            direction_trend = 'bear'
    except Exception:
        direction_trend = 'neutral'

    out['trend'] = {
        'scores': trend_scores_df,
        'intensity': trend_intensity,
        'direction': direction_trend
    }

    # --- Volatility indicators (5) ---
    vol_scores = {}
    atr14 = atr(df, length=14)
    vol_scores['ATR14'] = score_from_series(atr14)
    # realized vol: std(log returns) * sqrt(N)
    lr = np.log(close / close.shift()).dropna()
    realized_vol = lr.rolling(window=vol_lookback).std() * np.sqrt(vol_lookback)
    vol_scores['RealizedVol'] = score_from_series(realized_vol)
    # Bollinger bandwidth
    bbw = bollinger_bandwidth(close, length=20, ndev=2)
    vol_scores['BBWidth'] = score_from_series(bbw)
    # parkinson volatility
    pv = parkinson_volatility(df, length=vol_lookback)
    vol_scores['Parkinson'] = score_from_series(pv)
    # ATR ratio (ATR / mean close)
    vol_scores['ATR_ratio'] = score_from_series(atr_ratio(df, length=14))

    vol_scores_df = pd.Series(vol_scores)
    vol_intensity = vol_scores_df.apply(score_to_intensity)
    out['volatility'] = {
        'scores': vol_scores_df,
        'intensity': vol_intensity
    }

    # --- Momentum indicators (5) ---
    mom_scores = {}
    mom_scores['RSI14_dist'] = score_from_series((rsi(close, length=14) - 50).abs() / 50.0)
    mom_scores['ROC12_abs'] = score_from_series(roc(close, lookback=12).abs())
    k, d = stochastic_k(df, k_period=14, d_period=3)
    mom_scores['StochK'] = score_from_series(k)
    mom_scores['MACD_hist_abs'] = score_from_series(macd_hist.abs().rolling(window=mom_lookback, min_periods=1).mean())
    mom_scores['Price-EMA9'] = score_from_series(((close - ema(close, span=9)).abs() / (close + 1e-12)))

    mom_scores_df = pd.Series(mom_scores)
    mom_intensity = mom_scores_df.apply(score_to_intensity)
    # direction momentum (sign)
    direction_mom = 'neutral'
    try:
        if rsi(close, length=14).iloc[-1] > 55:
            direction_mom = 'bull'
        elif rsi(close, length=14).iloc[-1] < 45:
            direction_mom = 'bear'
    except Exception:
        direction_mom = 'neutral'

    out['momentum'] = {
        'scores': mom_scores_df,
        'intensity': mom_intensity,
        'direction': direction_mom
    }

    # --- Volume indicators (5) --- (only if 'Volume' exists)
    if 'Volume' in df.columns and not df['Volume'].isna().all():
        volsec_scores = {}
        vol_series = df['Volume']
        volsec_scores['Vol/MA20'] = score_from_series(vol_series / (vol_series.rolling(20).mean() + 1e-12))
        obv_series = obv(df)
        volsec_scores['OBV_change'] = score_from_series(obv_series.diff().abs())
        volsec_scores['Vol_percentile'] = score_from_series(vol_series)  # raw percentile of volume
        volsec_scores['MFI14'] = score_from_series(money_flow_index(df, length=14).abs())  # abs to reflect intensity
        # volume spike: recent vol / mean vol
        volsec_scores['Vol_spike'] = score_from_series(vol_series / (vol_series.rolling(20).mean() + 1e-12))

        volsec_scores_df = pd.Series(volsec_scores)
        volsec_intensity = volsec_scores_df.apply(score_to_intensity)
        out['volume'] = {
            'scores': volsec_scores_df,
            'intensity': volsec_intensity
        }
    else:
        out['volume'] = None

    return out

if __name__ == "__main__":
    
    import yfinance as yf 
    
    df = yf.download("TSLA", period="max", interval="1d", multi_level_index=False)
    
    # compute panel and plot
    panel = compute_panel(df)
