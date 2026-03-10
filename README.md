[English](README.md) | [Español](README.es.md) | [Português](README.pt.md) | [中文](README.zh.md)

---

# 📊 Quant Terminal: HMM + Chronos + FinBERT

This project is a **High-Performance Quantitative Dashboard** designed for technical, predictive, and sentiment analysis of financial assets. It combines Deep Learning architectures, probabilistic Hidden Markov Models (HMM), and Natural Language Processing (NLP).

![Tool Demonstration](assets/video_muestra_1.gif)

---

## 🛠 Detailed Architecture & Methodology

To ensure full transparency in calculations, this terminal breaks down its methodology into three processing layers:

### 1. Regime Detection (Hidden Markov Models)
The HMM model segments the market based on the statistical structure of the data, rather than fixed analyst rules.

*   **Input Variables (Features):**
    *   `log_r`: Logarithmic Returns (captures continuous percentage changes).
    *   `range`: Intra-period Range (High/Low - 1), an indicator of immediate volatility.
    *   `abs_r`: Absolute value of the return (strength of the movement).
    *   `vol_5`: Short-term volatility (5-period Std Dev).
*   **Algorithm:** `GaussianHMM` with 3 components. States are trained using the **Expectation-Maximization (Baum-Welch)** algorithm.
*   **Automatic Alignment:** States are automatically mapped according to the mean return:
    *   **Bear:** State with the lowest mean return.
    *   **Bull:** State with the highest mean return.
    *   **Side:** Intermediate state.
*   **Walk-Forward Validation:** The model is periodically retrained (rolling window) to adapt to structural breaks in the market.

### 2. Probabilistic Prediction (Chronos)
**Chronos** is a Transformer architecture by Amazon designed to treat time series like a language.

![HMM and Chronos Model](assets/hmm_chronos.png)

*   **Methodology:** The price is quantized into tokens and the model predicts the probability distribution of the next value.
*   **Zero-Shot Learning:** Doesn't rely on classic patterns (like head-and-shoulders); understands large-scale intrinsic temporal dynamics.
*   **Uncertainty:** The shaded area on the chart represents confidence bands (10% and 90% quantiles). If the bands are narrow, the model has high confidence in the trajectory.

### 3. Institutional-Grade NLP (FinBERT)
Utilizes a **BERT (Bidirectional Encoder Representations from Transformers)** neural network pre-trained with millions of financial documents.

*   **Sentiment Gap Calculation:** 
    *   Probabilities for each class are extracted: `[Positive, Negative, Neutral]`.
    *   $\text{Gap} = (\text{Prob}_{\text{Pos}} - \text{Prob}_{\text{Neg}}) \times 100$.
    *   A value of **100** indicates absolute optimism, **-100** indicates absolute panic.

![Technical and Sentiment Analysis](assets/deep_analysis.png)

---

## 🌡️ Indicator Transparency (Heatmap)

The intensity heatmap uses the following set of indicators for decision-making:

| Category | Indicator | Base Calculation |
| :--- | :--- | :--- |
| **Momentum** | RSI (14) | Relative Strength Index (Wilder). |
| | ROC (12) | 12-period Rate of Change. |
| | Stochastic K | Stochastic Oscillator (14, 3). |
| | MACD Hist | Difference between the MACD line and its signal. |
| **Volatility** | ATR (14) | Average True Range. |
| | Realized Vol | Moving standard deviation of returns. |
| | BB Width | Bollinger Bands Width (normalized). |
| | Parkinson | Volatility based on High/Low (more sensitive than close-based). |
| **Trend** | EMA (20) | Fast Exponential Moving Average. |
| | ADX (14) | Average Directional Index (trend strength). |
| | Price vs EM | Price position relative to its moving average. |
| **Volume** | Vol/MA20 | Current volume vs 20-day average. |
| | OBV Change | On-Balance Volume variation. |
| | Vol Spike | Detection of unusual volume spikes. |

---

## 💡 Usage Strategies & Recommendations

*   **Technical Confluence:** Look for the "Triple Check": Bull Regime (HMM) + Bullish Projection (Chronos) + Sentiment Gap > 10 (FinBERT).
*   **Heatmap Interpretation:** A uniform green block in "Trend" and "Momentum" confirms a healthy trend. Red blocks in "Volatility" usually precede periods of calm.
*   **Risks:** AI models are probabilistic. Never use this terminal as your sole execution source without proper stop-loss management.

---

## 📄 Raw Data & Data Auditing

Total calculation transparency through access to the raw data used by the models and the asset's historical behavior.

![Data and Projections](assets/raw_data.png)

---

## 🚀 Installation

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run: `streamlit run quant_dashboard_streamlit_app.py`

---
**Disclaimer:** *This dashboard is a statistical analysis tool and does not constitute financial advice.*
