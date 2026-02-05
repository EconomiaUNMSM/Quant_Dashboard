# from transformers import pipeline
import yfinance as yf
# ya tienes: pandas as pd, numpy as np, logging, datetime, etc.

import hashlib
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Optional
# Transformers (FinBERT)
from transformers import pipeline

logger = logging.getLogger("news_sentiment_search")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

def init_finbert_pipeline(model_name: str = "yiyanghkust/finbert-tone", device: Optional[int] = None, verbose: bool = False):
    """
    Intenta inicializar transformers.pipeline para FinBERT (sentiment-analysis).
    Si falla devuelve None (usaremos mock).
    device: pasar 0/1.. para GPU si quieres y transformers lo soporta.
    """
    try:
        # `pipeline` fue importado por ti desde transformers; si no, esto fallará.
        # Pedimos return_all_scores=True para obtener probabilidades por etiqueta.
        kwargs = {"model": model_name, "tokenizer": model_name, "return_all_scores": True}
        if device is not None:
            kwargs["device"] = device
        nlp = pipeline("sentiment-analysis", **kwargs)
        if verbose:
            logger.info("FinBERT pipeline inicializado con modelo: %s", model_name)
        return nlp
    except Exception as e:
        logger.warning("No se pudo inicializar FinBERT pipeline (fallback a mock). Error: %s", e)
        return None


def _safe_parse_news_from_yf(search_result):
    """Normaliza la salida de yfinance.Search(...).news -> lista de dicts con title, link, datetime, type, publisher"""
    out = []
    if not search_result:
        return out
    for item in search_result:
        try:
            title = item.get("title", "") or ""
            link = item.get("link", "") or ""
            ts = item.get("providerPublishTime", None)
            if ts:
                try:
                    dt = datetime.fromtimestamp(int(ts))
                except Exception:
                    dt = None
            else:
                dt = None
            typ = item.get("type", "") or ""
            publisher = item.get("publisher", item.get("providerName", "")) or ""
            out.append({"title": title, "link": link, "datetime": dt, "type": typ, "publisher": publisher})
        except Exception:
            continue
    return out


def _mock_finbert_scores_for_title(title: str):
    """Mock determinista: genera scores reproducibles a partir del hash del título."""
    h = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(h % (2**31 - 1))
    pos = rng.uniform(0.0, 1.0)
    neg = rng.uniform(0.0, 1.0)
    neu = rng.uniform(0.0, 1.0)
    arr = np.array([pos, neu, neg], dtype=float)
    arr = arr / arr.sum()
    return {"positive": float(arr[0]), "neutral": float(arr[1]), "negative": float(arr[2])}


def finbert_score_batch(titles: List[str], nlp_finbert, batch_size: int = 8) -> List[dict]:
    """
    Aplica FinBERT (pipeline) en batches si `nlp_finbert` existe; si es None, usa mock determinista.
    Retorna lista de dicts con keys 'positive','neutral','negative' y valores floats (0..1).
    """
    results = []
    if not titles:
        return results

    if nlp_finbert is None:
        # mock para todos los títulos
        for t in titles:
            results.append(_mock_finbert_scores_for_title(t))
        return results

    # pipeline real: ejecutar en batches
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        try:
            preds = nlp_finbert(batch)  # se espera return_all_scores=True -> lista de listas
        except Exception as e:
            logger.warning("Error durante inferencia FinBERT para batch: %s. Usando mock para ese batch.", e)
            preds = None

        if preds is None:
            for t in batch:
                results.append(_mock_finbert_scores_for_title(t))
        else:
            # convertir cada pred (lista de scores) a dict label->score; labels podrían ser 'POSITIVE'/'NEGATIVE'/'NEUTRAL' o variantes
            for p in preds:
                try:
                    mapping = {}
                    for d in p:
                        lbl = d.get("label", str(d.get("label", "")).lower()).lower()
                        score = float(d.get("score", 0.0))
                        mapping[lbl] = score
                    # normalizar keys a positive/neutral/negative
                    results.append({
                        "positive": float(mapping.get("positive", mapping.get("pos", mapping.get("positive",'') or 0.0))),
                        "neutral": float(mapping.get("neutral", mapping.get("neu", 0.0))),
                        "negative": float(mapping.get("negative", mapping.get("neg", 0.0)))
                    })
                except Exception:
                    # fallback mock por si el formato no es el esperado
                    results.append(_mock_finbert_scores_for_title("fallback"))
    return results


def search_news_sentiment(query: str,
                          max_news: int = 10,
                          news_count: int = 20,
                          finbert_batch: int = 8,
                          finbert_model: str = "yiyanghkust/finbert-tone",
                          use_finbert_pipeline: bool = True,
                          device: Optional[int] = None) -> pd.DataFrame:
    """
    Busca noticias relacionadas a `query` y aplica FinBERT (o mock).
    Retorna DataFrame con columnas: Fecha, Hora, Titular, pos_%, neg_%, Gap
    Ordenado de más reciente a más antiguo.
    Parámetros:
      - max_news: máximo de noticias a retornar (algunos backends devuelven máximo 10).
      - news_count: parámetro pasado a yfinance.Search (cuántas noticias intentar recuperar).
      - finbert_batch: batch size para pipeline.
      - finbert_model: nombre del modelo para inicializar pipeline si use_finbert_pipeline=True.
      - device: opcional, entero para GPU device=0, etc.
    """
    # inicializar pipeline si se desea
    nlp = None
    if use_finbert_pipeline:
        try:
            nlp = init_finbert_pipeline(model_name=finbert_model, device=device, verbose=False)
        except Exception as e:
            logger.warning("Error inicializando FinBERT pipeline, se usará mock: %s", e)
            nlp = None

    # 1) obtener noticias via yfinance.Search (si falla, fallback a mock)
    news_items = []
    try:
        # yfinance debe estar importado en el entorno
        res = yf.Search(query=query, news_count=news_count)
        raw = getattr(res, "news", []) or []
        news_items = _safe_parse_news_from_yf(raw)
    except Exception as e:
        logger.info("yfinance.Search falló o no disponible -> generando noticias mock. Error: %s", e)
        news_items = []

    # fallback mock si no obtuvimos resultados reales
    if not news_items:
        now = datetime.utcnow()
        mock = []
        for i in range(min(max_news, news_count)):
            dt = now - timedelta(minutes=15 * i)
            mock.append({"title": f"Mock news about {query} #{i+1}",
                         "link": "",
                         "datetime": dt,
                         "type": "news",
                         "publisher": "MockPublisher"})
        news_items = mock

    # limitar a max_news
    if max_news is not None and max_news > 0:
        news_items = news_items[:max_news]

    # extraer títulos y timestamps
    titles = [it.get("title", "") for it in news_items]
    datetimes = [it.get("datetime", None) for it in news_items]

    # 2) aplicar FinBERT (o mock) en batches
    probs = finbert_score_batch(titles, nlp_finbert=nlp, batch_size=finbert_batch)

    # 3) construir DataFrame con las columnas solicitadas
    rows = []
    for t, dt, p in zip(titles, datetimes, probs):
        try:
            dt_parsed = pd.to_datetime(dt) if dt is not None else pd.NaT
        except Exception:
            dt_parsed = pd.NaT
        pos = float(p.get("positive", 0.0))
        neg = float(p.get("negative", 0.0))
        pos_pct = round(pos * 100.0, 2)
        neg_pct = round(neg * 100.0, 2)
        gap = round(pos_pct - neg_pct, 2)
        rows.append({"datetime": dt_parsed, "Titular": t, "pos_%": pos_pct, "neg_%": neg_pct, "Gap": gap})

    df = pd.DataFrame(rows)

    # ordenar de más reciente a más antiguo (NaT van al final)
    if not df.empty:
        df = df.sort_values(by="datetime", ascending=False, na_position="last").reset_index(drop=True)
        df["Fecha"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["Hora"] = df["datetime"].dt.strftime("%H:%M:%S")
    else:
        df["Fecha"] = []
        df["Hora"] = []

    # columnas exactas solicitadas, en el orden pedido
    cols = ["Fecha", "Hora", "Titular", "pos_%", "neg_%", "Gap"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    df_out = df[cols].copy()
    return df_out


if __name__ == "__main__":
    
    # Prueba
    df = search_news_sentiment(query="DECK")
    
