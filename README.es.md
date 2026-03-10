[English](README.md) | [Español](README.es.md) | [Português](README.pt.md) | [中文](README.zh.md)

---

# 📊 Quant Terminal: HMM + Chronos + FinBERT

Este proyecto es un **Dashboard Cuantitativo de Alto Rendimiento** diseñado para el análisis técnico, predictivo y de sentimiento de activos financieros. Combina arquitecturas de aprendizaje profundo (Deep Learning), modelos probabilísticos de estados ocultos (HMM) y procesamiento de lenguaje natural (NLP).

![Demostración de la Herramienta](assets/video_muestra_1.gif)

---

## 🛠 Arquitectura y Metodología Detallada

Para garantizar la transparencia en los cálculos, este terminal desglosa su metodología en tres capas de procesamiento:

### 1. Detección de Regímenes (Hidden Markov Models)
El modelo HMM segmenta el mercado basándose en la estructura estadística de los datos, no en reglas fijas de analistas.

*   **Variables de Entrada (Features):**
    *   `log_r`: Retornos Logarítmicos (captura cambios porcentuales continuos).
    *   `range`: Rango Intra-periodo (High/Low - 1), indicador de volatilidad inmediata.
    *   `abs_r`: Valor absoluto del retorno (fuerza del movimiento).
    *   `vol_5`: Volatilidad de corto plazo (Std Dev de 5 periodos).
*   **Algoritmo:** `GaussianHMM` con 3 componentes. Los estados se entrenan mediante el algoritmo de **Expectation-Maximization (Baum-Welch)**.
*   **Alineación Automática:** Los estados se mapean automáticamente según el retorno medio:
    *   **Bear (Bajista):** Estado con el retorno medio más bajo.
    *   **Bull (Alcista):** Estado con el retorno medio más alto.
    *   **Side (Lateral):** Estado intermedio.
*   **Validación Walk-Forward:** El modelo se re-entrena periódicamente (ventana móvil) para adaptarse a cambios estructurales ("Structural Breaks") en el mercado.

### 2. Predicción Probabilística (Chronos)
**Chronos** es una arquitectura Transformer de Amazon diseñada para tratar las series temporales como un lenguaje.

![Modelo HMM y Chronos](assets/hmm_chronos.png)

*   **Metodología:** El precio se cuantiza en tokens y el modelo predice la distribución de probabilidad del siguiente valor.
*   **Zero-Shot Learning:** No depende de patrones clásicos (como cabeza-hombros); entiende la dinámica temporal intrínseca a gran escala.
*   **Incertidumbre:** El área sombreada en el gráfico representa las bandas de confianza (cuantiles 10% y 90%). Si las bandas son estrechas, el modelo tiene alta confianza en la trayectoria.

### 3. NLP de Grado Institucional (FinBERT)
Utiliza una red neuronal **BERT (Bidirectional Encoder Representations from Transformers)** pre-entrenada con millones de documentos financieros.

*   **Cálculo del Sentiment Gap:** 
    *   Se extraen las probabilidades para cada clase: `[Positivo, Negativo, Neutral]`.
    *   $\text{Gap} = (\text{Prob}_{\text{Pos}} - \text{Prob}_{\text{Neg}}) \times 100$.
    *   Un valor de **100** indica optimismo absoluto, **-100** indica pánico absoluto.

![Análisis Técnico y de Sentimiento](assets/deep_analysis.png)

---

## 🌡️ Transparencia de Indicadores (Heatmap)

El mapa de calor de intensidad utiliza el siguiente set de indicadores para la toma de decisiones:

| Categoría | Indicador | Cálculo Base |
| :--- | :--- | :--- |
| **Momentum** | RSI (14) | Índice de Fuerza Relativa (Wilder). |
| | ROC (12) | Rate of Change de 12 periodos. |
| | Stochastic K | Oscilador Estocástico (14, 3). |
| | MACD Hist | Diferencia entre la línea MACD y su señal. |
| **Volatility** | ATR (14) | Average True Range. |
| | Realized Vol | Desviación estándar móvil de los retornos. |
| | BB Width | Ancho de las Bandas de Bollinger (normalizado). |
| | Parkinson | Volatilidad basada en High/Low (más sensible que la de cierre). |
| **Trend** | EMA (20) | Media Móvil Exponencial rápida. |
| | ADX (14) | Average Directional Index (fuerza de la tendencia). |
| | Price vs EM | Posición del precio respecto a su media. |
| **Volume** | Vol/MA20 | Volumen actual vs promedio de 20 días. |
| | OBV Change | Variación del On-Balance Volume. |
| | Vol Spike | Detección de picos inusuales de volumen. |

---

## 💡 Estrategias de Uso y Recomendaciones

*   **Confluencia Técnica:** Busque el "Triple Check": Régimen Bull (HMM) + Proyección alcista (Chronos) + Sentiment Gap > 10 (FinBERT).
*   **Interpretación del Heatmap:** Un bloque verde uniforme en "Trend" y "Momentum" confirma una tendencia saludable. Los bloques rojos en "Volatility" suelen preceder a periodos de calma.
*   **Riesgos:** Los modelos de IA son probabilísticos. Nunca utilice este terminal como única fuente de ejecución sin una gestión de stop-loss adecuada.

---

## 📄 Raw Data y Auditoría de Datos

Transparencia total en los cálculos mediante el acceso a los datos crudos utilizados por los modelos y el histórico del activo.

![Datos y Proyecciones](assets/raw_data.png)

---

## 🚀 Instalación

1.  Instala las dependencias: `pip install -r requirements.txt`
2.  Ejecuta: `streamlit run quant_dashboard_streamlit_app.py`

---
**Disclaimer:** *Este dashboard es una herramienta de análisis estadístico y no constituye una asesoría financiera.*
