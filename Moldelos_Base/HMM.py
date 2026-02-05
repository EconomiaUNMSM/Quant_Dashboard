import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import math

# -------------------------
# Utilidades
# -------------------------

def build_features(df):
    df = df.copy()
    df['log_r'] = np.log(df['Close'] / df['Close'].shift(1))
    df['range'] = df['High'] / df['Low'] - 1.0
    df['abs_r'] = df['log_r'].abs()
    df['vol_5'] = df['log_r'].rolling(5).std()
    df = df.dropna()
    X = df[['log_r','range','abs_r','vol_5']].copy()
    return df, X

# -------------------------
# Fit HMM robusto
# -------------------------

def fit_hmm(X, n_states=3, covariance_type='full', random_state=42, n_iter=500):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type=covariance_type,
                            n_iter=n_iter,
                            init_params='stmc',
                            random_state=random_state,
                            verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xs)
    post = model.predict_proba(Xs)
    viterbi = model.predict(Xs)
    return model, scaler, viterbi, post

# -------------------------
# Mapear estados por estadísticos (entrenamiento)
# -------------------------

def map_states_by_train_stats(df_features, assigned_states):
    ser = pd.Series(assigned_states, index=df_features.index, name='state')
    stats = []
    for s in np.unique(assigned_states):
        mask = ser == s
        mean_r = df_features.loc[mask, 'log_r'].mean()
        std_r = df_features.loc[mask, 'log_r'].std()
        stats.append((int(s), mean_r, std_r, int(mask.sum())))
    stats_df = pd.DataFrame(stats, columns=['state','mean_r','std_r','count']).set_index('state')
    sorted_states = stats_df.sort_values('mean_r').index.tolist()
    mapping = {}
    if len(sorted_states) >= 3:
        low = sorted_states[0]
        high = sorted_states[-1]
        mapping[low] = 'bear'
        mapping[high] = 'bull'
        for s in sorted_states[1:-1]:
            mapping[s] = 'side'
    else:
        for i, s in enumerate(sorted_states):
            if i == 0:
                mapping[s] = 'bear'
            elif i == len(sorted_states)-1:
                mapping[s] = 'bull'
            else:
                mapping[s] = 'side'
    labeled = pd.Series([mapping.get(int(s),'side') for s in assigned_states], index=df_features.index)
    return mapping, labeled, stats_df

# -------------------------
# Asignación usando predict_proba + umbral + hysteresis
#    retorna (labels_series, prob_df)
# -------------------------

def assign_states_with_probabilities(posteriors, mapping_train, prob_threshold=0.6, index=None, verbose=False):
    T, n_states = posteriors.shape
    state_idx_to_label = {int(k): v for k, v in mapping_train.items()}
    labels = []
    prev_label = 'side'
    label_to_states = {}
    for s_idx, lbl in state_idx_to_label.items():
        label_to_states.setdefault(lbl, []).append(int(s_idx))
    rows = []
    for t in range(T):
        probs = posteriors[t].astype(float)
        arg = int(np.argmax(probs))
        maxp = float(probs[arg])
        mapped_label = state_idx_to_label.get(arg, 'side')
        label_probs = {lbl: float(np.sum(probs[[int(i) for i in idxs]])) for lbl, idxs in label_to_states.items()}
        if maxp >= prob_threshold:
            label = mapped_label
            action = 'assign'
            prev_label = label
        else:
            label = prev_label
            action = 'hold_prev'
        labels.append(label)
        row = {
            't': t,
            'state_idx': arg,
            'mapped_label': mapped_label,
            'assigned_label': label,
            'max_p': maxp,
            'action': action,
        }
        for lbl in ['bull','side','bear']:
            row[f'prob_{lbl}'] = float(label_probs.get(lbl, 0.0))
        for i in range(n_states):
            row[f'p_state_{i}'] = float(probs[i])
        rows.append(row)
        if verbose:
            ts = None
            if index is not None:
                try:
                    ts = index[t]
                except Exception:
                    ts = None
            if ts is not None:
                print(f"{pd.to_datetime(ts).date()} | st={arg} ({mapped_label}) | max_p={maxp:.3f} | action={action} | agg={{'bull':{row['prob_bull']:.3f}, 'side':{row['prob_side']:.3f}, 'bear':{row['prob_bear']:.3f}}}")
            else:
                print(f"t={t} | st={arg} ({mapped_label}) | max_p={maxp:.3f} | action={action} | agg={{'bull':{row['prob_bull']:.3f}, 'side':{row['prob_side']:.3f}, 'bear':{row['prob_bear']:.3f}}}")
    prob_df = pd.DataFrame(rows)
    if index is not None:
        try:
            prob_df.index = pd.to_datetime(index)
        except Exception:
            pass
    labels_s = pd.Series(labels, index=prob_df.index if index is not None else None)
    return labels_s, prob_df

# -------------------------
# Suavizado mínimo de duración
# -------------------------

def enforce_min_duration(state_series, min_run=3):
    arr = state_series.values.copy().astype(object)
    idx = 0
    n = len(arr)
    runs = []
    while idx < n:
        val = arr[idx]
        j = idx + 1
        while j < n and arr[j] == val:
            j += 1
        runs.append((idx, j, val))
        idx = j
    for i, (start, end, val) in enumerate(runs):
        length = end - start
        if length < min_run:
            if i > 0:
                prev_val = runs[i-1][2]
                arr[start:end] = prev_val
            elif i < len(runs)-1:
                next_val = runs[i+1][2]
                arr[start:end] = next_val
    return pd.Series(arr, index=state_series.index)

# -------------------------
# Signals & backtest
# -------------------------

def build_positions_from_labels(labels):
    mapping = {'bull': 1, 'bear': -1, 'side': 0}
    pos = labels.map(mapping).shift(1).fillna(0).astype(int)
    return pos


def backtest_strategy(df_close, positions, transaction_cost=0.000, verbose=False):
    if isinstance(df_close, pd.DataFrame):
        if 'Close' in df_close.columns:
            prices = df_close['Close'].copy()
        elif df_close.shape[1] == 1:
            prices = df_close.iloc[:, 0].copy()
        else:
            prices = df_close.iloc[:, 0].copy()
    else:
        prices = df_close.copy()
    if isinstance(positions, pd.DataFrame):
        if positions.shape[1] == 1:
            pos = positions.iloc[:, 0].copy()
        else:
            pos = positions.iloc[:, 0].copy()
    else:
        pos = positions.copy()
    try:
        prices = prices.reindex(pos.index)
    except Exception:
        prices = pd.Series(prices.values.flatten(), index=pos.index)
    prices = pd.to_numeric(prices, errors='coerce')
    pos = pd.to_numeric(pos, errors='coerce')
    valid_mask = prices.notna() & pos.notna()
    if not valid_mask.all():
        prices = prices.loc[valid_mask]
        pos = pos.loc[valid_mask]
    if len(prices) == 0 or len(pos) == 0:
        empty_series = pd.Series(dtype=float)
        results = {
            'returns': empty_series,
            'strat_ret': empty_series,
            'strat_ret_net': empty_series,
            'cum': empty_series,
            'cum_bh': empty_series,
            'total_return': 0.0,
            'total_return_bh': 0.0,
            'ann_ret': np.nan,
            'ann_vol': np.nan,
            'sharpe': np.nan,
            'max_dd': np.nan,
            'trades': 0,
            'win_rate': np.nan
        }
        if verbose:
            print("backtest_strategy: no data after alignment -> returning empty metrics.")
        return results
    prices = pd.Series(prices.values.flatten(), index=prices.index).astype(float)
    pos = pd.Series(pos.values.flatten(), index=pos.index).astype(float)
    returns = prices.pct_change().fillna(0).astype(float)
    strat_ret = pos * returns
    turnover = pos.diff().abs().fillna(0)
    costs = turnover * transaction_cost
    strat_ret_net = strat_ret - costs
    cum = (1 + strat_ret_net).cumprod()
    if len(prices) > 0:
        cum_bh = prices / prices.iloc[0]
    else:
        cum_bh = pd.Series(dtype=float)
    def last_scalar_from_series(s):
        if s is None:
            return 0.0
        arr = np.asarray(s)
        if arr.size == 0:
            return 0.0
        last = arr[-1]
        try:
            return float(np.asarray(last).flatten()[0])
        except Exception:
            return float(last)
    total_return = last_scalar_from_series(cum) - 1.0
    total_return_bh = last_scalar_from_series(cum_bh) - 1.0
    days = float(len(returns)) if len(returns) > 0 else 0.0
    ann_ret = (1 + total_return) ** (252.0 / days) - 1.0 if (days > 0 and not np.isnan(total_return)) else np.nan
    ann_vol = float(np.nanstd(strat_ret_net.values.flatten()) * np.sqrt(252.0)) if len(strat_ret_net) > 1 else np.nan
    sharpe = float(ann_ret / ann_vol) if (ann_vol is not None and not np.isnan(ann_vol) and ann_vol > 0) else np.nan
    if len(cum) > 0:
        running_max = cum.cummax()
        drawdown = (cum / running_max) - 1.0
        max_dd = float(drawdown.min())
    else:
        max_dd = np.nan
    trades = int((turnover > 0).sum())
    trade_returns = strat_ret_net[turnover > 0]
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) > 0 else np.nan
    results = {
        'returns': returns,
        'strat_ret': strat_ret,
        'strat_ret_net': strat_ret_net,
        'cum': cum,
        'cum_bh': cum_bh,
        'total_return': float(total_return),
        'total_return_bh': float(total_return_bh),
        'ann_ret': float(ann_ret) if not np.isnan(ann_ret) else np.nan,
        'ann_vol': float(ann_vol) if not np.isnan(ann_vol) else np.nan,
        'sharpe': float(sharpe) if not np.isnan(sharpe) else np.nan,
        'max_dd': float(max_dd) if not np.isnan(max_dd) else np.nan,
        'trades': trades,
        'win_rate': float(win_rate) if not np.isnan(win_rate) else np.nan
    }
    if verbose:
        print(f"Total strat ret: {results['total_return']:.6f}, BH: {results['total_return_bh']:.6f}, AnnRet: {results['ann_ret']:.6f}, Sharpe: {results['sharpe']:.3f}, MaxDD: {results['max_dd']:.3f}, Trades: {results['trades']}, WinRate: {results['win_rate']:.3f}")
    return results

# -------------------------
# Walk-forward evaluation (rolling) - export CSV opcional
# -------------------------

def walk_forward_hmm(df_feat, X, n_states=3,
                      train_window=252*2, test_window=63, step_size=None, step=None,
                      prob_threshold=0.6, min_run=3, transaction_cost=0.0005,
                      verbose=True, print_probs=False, save_probs_csv=None):
    if step_size is None:
        step_size = test_window if step is None else step
    N = len(X)
    fold_results = []
    fold_dfs = []
    prob_dfs = []
    fold_i = 0
    for start in range(0, N - train_window - test_window + 1, step_size):
        fold_i += 1
        train_idx = range(start, start + train_window)
        test_idx = range(start + train_window, start + train_window + test_window)
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        df_train = df_feat.iloc[train_idx]
        df_test = df_feat.iloc[test_idx]
        try:
            model, scaler, viterbi_train, post_train = fit_hmm(X_train, n_states=n_states)
        except Exception as e:
            if verbose:
                print(f"[Fold {fold_i}] HMM fit failed: {e}")
            continue
        mapping_train, labeled_train, stats_train = map_states_by_train_stats(df_train, viterbi_train)
        try:
            X_test_s = scaler.transform(X_test)
        except Exception as e:
            if verbose:
                print(f"[Fold {fold_i}] scaler.transform failed: {e}")
            continue
        post_test = model.predict_proba(X_test_s)
        labels_test, prob_df = assign_states_with_probabilities(post_test, mapping_train, prob_threshold=prob_threshold, index=df_test.index, verbose=print_probs)
        prob_df = prob_df.copy()
        prob_df['fold'] = fold_i
        prob_dfs.append(prob_df)
        if isinstance(labels_test, pd.Series):
            labels_test_s = labels_test.copy()
        else:
            labels_test_arr = np.asarray(labels_test).flatten()
            labels_test_s = pd.Series(labels_test_arr, index=df_test.index)
        if not labels_test_s.index.equals(df_test.index):
            labels_test_s.index = df_test.index
        labels_test_s = enforce_min_duration(labels_test_s, min_run=min_run)
        positions = build_positions_from_labels(labels_test_s)
        positions_s = pd.Series(np.asarray(positions).flatten(), index=df_test.index)
        close = df_test['Close'].copy()
        if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
            close = close.iloc[:, 0]
        fold_df = pd.DataFrame({'close': close, 'pos': positions_s, 'label': labels_test_s})
        fold_df = fold_df.dropna(subset=['close'])
        if fold_df.empty:
            if verbose:
                print(f"[Fold {fold_i}] fold_df vacío tras dropna -> skipping.")
            continue
        try:
            results = backtest_strategy(fold_df['close'], fold_df['pos'], transaction_cost=transaction_cost, verbose=False)
        except Exception as e:
            if verbose:
                print(f"[Fold {fold_i}] backtest fallo: {e}")
            continue
        if verbose:
            print(f"Fold {fold_i} [{fold_df.index[0].date()} - {fold_df.index[-1].date()}] -> Return: {results['total_return']:.6f}, Sharpe: {results['sharpe']:.3f}, Trades: {results['trades']}")
        fold_results.append({
            'fold': fold_i,
            'start': fold_df.index[0],
            'end': fold_df.index[-1],
            'mapping_train': mapping_train,
            'stats_train': stats_train,
            'results': results,
            'labels_test': labels_test_s,
            'positions': fold_df['pos']
        })
        fold_dfs.append(fold_df[['close','pos','label']])
    if len(fold_dfs) == 0:
        if verbose:
            print("No se generaron folds válidos en walk-forward.")
        return {'folds': fold_results, 'aggregated': None, 'positions': pd.Series(dtype=float), 'close': pd.Series(dtype=float), 'labels': pd.Series(dtype=object), 'probabilities': pd.DataFrame()}
    aggregated_df = pd.concat(fold_dfs)
    aggregated_df = aggregated_df[~aggregated_df.index.duplicated(keep='first')].sort_index()
    aggregated_df['pos'] = aggregated_df['pos'].astype(float).values.flatten()
    aggregated_df['close'] = aggregated_df['close'].astype(float).values.flatten()
    aggregated_df['label'] = aggregated_df['label'].astype(str).values.flatten()
    aggregated_positions = pd.Series(aggregated_df['pos'].values, index=aggregated_df.index)
    aggregated_close = pd.Series(aggregated_df['close'].values, index=aggregated_df.index)
    aggregated_labels = pd.Series(aggregated_df['label'].values, index=aggregated_df.index)
    if not aggregated_positions.empty and not aggregated_close.empty:
        agg_bt = backtest_strategy(aggregated_close, aggregated_positions, transaction_cost=transaction_cost, verbose=False)
        if verbose:
            print("=== AGGREGATED OOS PERFORMANCE ===")
            print(f"Total Return: {agg_bt['total_return']:.6f}, Sharpe: {agg_bt['sharpe']:.3f}, Trades: {agg_bt['trades']}, WinRate: {agg_bt['win_rate']:.3f}")
    else:
        agg_bt = None
        if verbose:
            print("No hay datos para backtest agregado (aggregated_positions/close vacíos).")
    probs_all = pd.concat(prob_dfs) if len(prob_dfs) > 0 else pd.DataFrame()
    if save_probs_csv is not None and not probs_all.empty:
        try:
            probs_all.to_csv(save_probs_csv, index=True)
            if verbose:
                print(f"Probabilities exported to {save_probs_csv}")
        except Exception as e:
            if verbose:
                print(f"Error saving probabilities CSV: {e}")
    return {'folds': fold_results, 'aggregated': agg_bt, 'positions': aggregated_positions, 'close': aggregated_close, 'labels': aggregated_labels, 'probabilities': probs_all}

if __name__ == "__main__":
    
    # Importar datos - Importancia de labels para la gráfica principal
    df = yf.download("BTC-USD", period="max", interval="5m", multi_level_index=False, ignore_tz=False)
    
    # Características
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
    
    # Grafico de los Estados
    labels = wf['labels'].reindex(df.index).fillna(method='ffill').fillna('side')

