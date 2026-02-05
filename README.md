# 📊 Quant Terminal: HMM + Chronos + FinBERT

Este proyecto es un **Dashboard Cuantitativo de Alto Rendimiento** diseñado para el análisis técnico, predictivo y de sentimiento de activos financieros. Combina arquitecturas de aprendizaje profundo (Deep Learning), modelos probabilísticos de estados ocultos (HMM) y procesamiento de lenguaje natural (NLP) para ofrecer una visión 360° del mercado.

---

## 🛠 Arquitectura y Metodología

El terminal se basa en tres pilares fundamentales que operan de forma independiente pero integrada:

### 1. Detección de Regímenes con HMM (Hidden Markov Models)
Utiliza la librería `hmmlearn` para segmentar el comportamiento del mercado en tres estados latentes (no observables directamente):

*   **Bull (Alcista):** Periodos de retornos positivos y baja/moderada volatilidad.
*   **Bear (Bajista):** Periodos de retornos negativos y alta volatilidad.
*   **Side (Lateral):** Periodos de consolidación o indecisión.
*   **Metodología:** El modelo se entrena mediante el algoritmo de *Expectation-Maximization (Baum-Welch)* utilizando retornos logarítmicos, volatilidad histórica y volumen como variables de entrada (features). Se aplica un *Walk-Forward Validation* para evitar el sesgo de supervivencia y el sobreajuste.

### 2. Predicción de Series Temporales con Chronos
Implementa **Chronos (Amazon)**, un modelo de lenguaje pre-entrenado adaptado específicamente para series temporales (TimeSeries Transformers).

*   **Funcionamiento:** Chronos trata los valores de precios como "tokens" de un lenguaje, permitiendo realizar predicciones de *Zero-Shot* (sin necesidad de entrenamiento específico para el ticker actual).
*   **Salida:** Proporciona una mediana de predicción y bandas de confianza (cuantiles) para los próximos $N$ periodos, capturando la incertidumbre intrínseca del pronóstico.

### 3. Análisis de Sentimiento con FinBERT
Aprovecha un modelo **BERT especializado en finanzas** para procesar noticias de última hora recopiladas vía Yahoo Finance/RSS.

*   **Valores de Sentimiento:**
    *   `pos_%`: Probabilidad de que la noticia sea favorable para el activo.
    *   `neg_%`: Probabilidad de impacto negativo.
    *   `Gap`: La diferencia neta (`pos` - `neg`). Un Gap > 0.5 indica un sentimiento extremadamente alcista, mientras que < -0.5 indica pánico o riesgo inminente.

---

## 🌡️ Panel de Indicadores (Heatmap de Intensidad)

El mapa de calor no solo muestra el valor del indicador, sino su **Intensidad Relativa (1 a 5)**:

*   **Metodología:** Los indicadores (RSI, Bandas de Bollinger, MACD, etc.) se normalizan y comparan con sus rangos históricos.
*   **Interpretación:**
    *   🟥 **1-2 (Baja):** Sobrecompra extrema o agotamiento de tendencia.
    *   🟨 **3 (Media):** Neutralidad o transición.
    *   🟩 **4-5 (Alta):** Fuerte impulso o señales de confirmación de tendencia.

---

## 🚀 Instalación y Uso

### Requisitos Previos
*   Python 3.10+
*   Entorno virtual recomendado (`venv`)

### Configuración
1.  Clona el repositorio.
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ejecuta la interfaz:
    ```bash
    streamlit run quant_dashboard_streamlit_app.py
    ```

---

## 📈 Ejemplo de Interpretación Técnica

**Escenario:**
*   **HMM:** Detecta una transición de *Side* a *Bull*.
*   **Chronos:** La mediana apunta a un crecimiento del 2% en las próximas 5 barras con bandas de confianza estrechas.
*   **FinBERT:** Gap positivo de 0.4 basado en las últimas noticias de ganancias por acción (EPS).
*   **Heatmap:** El grupo de *Momentum* muestra intensidades de 4 y 5.

**Conclusión:** Existe una convergencia de datos (confluencia) que sugiere una alta probabilidad de continuación alcista confirmada por fundamentales (sentimiento) y estructura de mercado (HMM).

---

## 📁 Estructura del Proyecto

*   `Indicadores/`: Lógica matemática y cálculo de señales técnicas.
*   `Moldelos_Base/`: Implementación de HMM, Chronos y FinBERT.
*   `Graficos/`: Funciones de visualización interactiva (Plotly) y estática (Matplotlib).
*   `Fuente_Datos/`: Módulos de conexión con APIs financieras (yfinance).

---
**Disclaimer:** *Este dashboard es una herramienta de análisis estadístico y no constituye una asesoría financiera. El trading de activos implica un alto riesgo de pérdida de capital.*
