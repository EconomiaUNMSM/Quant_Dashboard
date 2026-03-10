[English](README.md) | [Español](README.es.md) | [Português](README.pt.md) | [中文](README.zh.md)

---

# 📊 Terminal Quant: HMM + Chronos + FinBERT

Este projeto é um **Dashboard Quantitativo de Alto Desempenho** projetado para análise técnica, preditiva e de sentimento de ativos financeiros. Ele combina arquiteturas de Deep Learning, Modelos Ocultos de Markov probabilísticos (HMM) e Processamento de Linguagem Natural (NLP).

![Demonstração da Ferramenta](assets/video_muestra_1.gif)

---

## 🛠 Arquitetura e Metodologia Detalhada

Para garantir a transparência total nos cálculos, este terminal divide sua metodologia em três camadas de processamento:

### 1. Detecção de Regimes (Hidden Markov Models)
O modelo HMM segmenta o mercado com base na estrutura estatística dos dados, em vez de regras fixas de analistas.

*   **Variáveis de Entrada (Features):**
    *   `log_r`: Retornos Logarítmicos (captura mudanças percentuais contínuas).
    *   `range`: Intervalo Intra-período (High/Low - 1), indicador de volatilidade imediata.
    *   `abs_r`: Valor absoluto do retorno (força do movimento).
    *   `vol_5`: Volatilidade de curto prazo (Desvio Padrão de 5 períodos).
*   **Algoritmo:** `GaussianHMM` com 3 componentes. Os estados são treinados usando o algoritmo de **Expectation-Maximization (Baum-Welch)**.
*   **Alinhamento Automático:** Os estados são mapeados automaticamente de acordo com o retorno médio:
    *   **Bear (Baixa):** Estado com o menor retorno médio.
    *   **Bull (Alta):** Estado com o maior retorno médio.
    *   **Side (Lateral):** Estado intermediário.
*   **Validação Walk-Forward:** O modelo é retreinado periodicamente (janela móvel) para se adaptar a quebras estruturais no mercado.

### 2. Previsão Probabilística (Chronos)
**Chronos** é uma arquitetura Transformer da Amazon projetada para tratar séries temporais como uma linguagem.

![Modelo HMM e Chronos](assets/hmm_chronos.png)

*   **Metodologia:** O preço é quantizado em tokens e o modelo prevê a distribuição de probabilidade do próximo valor.
*   **Zero-Shot Learning:** Não depende de padrões clássicos (como ombro-cabeça-ombro); compreende a dinâmica temporal intrínseca em larga escala.
*   **Incerteza:** A área sombreada no gráfico representa as faixas de confiança (quantis de 10% e 90%). Se as faixas forem estreitas, o modelo tem alta confiança na trajetória.

### 3. NLP de Nível Institucional (FinBERT)
Utiliza uma rede neural **BERT (Bidirectional Encoder Representations from Transformers)** pré-treinada com milhões de documentos financeiros.

*   **Cálculo do Sentiment Gap:** 
    *   As probabilidades para cada classe são extraídas: `[Positivo, Negativo, Neutro]`.
    *   $\text{Gap} = (\text{Prob}_{\text{Pos}} - \text{Prob}_{\text{Neg}}) \times 100$.
    *   Um valor de **100** indica otimismo absoluto, **-100** indica pânico absoluto.

![Análise Técnica e de Sentimento](assets/deep_analysis.png)

---

## 🌡️ Transparência de Indicadores (Heatmap)

O mapa de calor (heatmap) de intensidade usa o seguinte conjunto de indicadores para tomada de decisões:

| Categoria | Indicador | Cálculo Base |
| :--- | :--- | :--- |
| **Momentum** | RSI (14) | Índice de Força Relativa (Wilder). |
| | ROC (12) | Rate of Change de 12 períodos. |
| | Stochastic K | Oscilador Estocástico (14, 3). |
| | MACD Hist | Diferença entre a linha MACD e seu sinal. |
| **Volatility** | ATR (14) | Average True Range. |
| | Realized Vol | Desvio padrão móvel dos retornos. |
| | BB Width | Largura das Bandas de Bollinger (normalizada). |
| | Parkinson | Volatilidade baseada no High/Low (mais sensível que o fechamento). |
| **Trend** | EMA (20) | Média Móvel Exponencial rápida. |
| | ADX (14) | Average Directional Index (força da tendência). |
| | Price vs EM | Posição do preço em relação à sua média móvel. |
| **Volume** | Vol/MA20 | Volume atual vs média de 20 dias. |
| | OBV Change | Variação do On-Balance Volume. |
| | Vol Spike | Detecção de picos incomuns de volume. |

---

## 💡 Estratégias de Uso e Recomendações

*   **Confluência Técnica:** Busque o "Check Triplo": Regime Bull (HMM) + Projeção de Alta (Chronos) + Sentiment Gap > 10 (FinBERT).
*   **Interpretação do Heatmap:** Um bloco verde uniforme em "Trend" e "Momentum" confirma uma tendência saudável. Blocos vermelhos em "Volatility" geralmente precedem períodos de calmaria.
*   **Riscos:** Os modelos de IA são probabilísticos. Nunca use este terminal como sua única fonte de execução sem um gerenciamento de stop-loss adequado.

---

## 📄 Dados Brutos (Raw Data) e Auditoria

Transparência total nos cálculos através do acesso aos dados brutos usados pelos modelos e o histórico do ativo.

![Dados e Projeções](assets/raw_data.png)

---

## 🚀 Instalação

1.  Instale as dependências: `pip install -r requirements.txt`
2.  Execute: `streamlit run quant_dashboard_streamlit_app.py`

---
**Aviso Legal:** *Este dashboard é uma ferramenta de análise estatística e não constitui aconselhamento financeiro.*
