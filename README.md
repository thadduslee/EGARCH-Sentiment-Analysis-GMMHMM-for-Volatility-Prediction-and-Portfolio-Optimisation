# Financial Portfolio Analysis & Optimization Pipeline

A comprehensive machine learning pipeline for quantitative finance that integrates social sentiment, asymmetric volatility modeling, market regime detection, and deep learning to optimize portfolio weights and forecast long-term wealth paths.

## Overview
This repository contains an end-to-end data engineering and quantitative modeling pipeline. It ingests stock data, scrapes and filters alternative data from financial subreddits, extracts signals using a fine-tuned Large Language Model, and models complex market dynamics using hybrid statistical and deep learning approaches (GMM-HMM, EGARCH, LSTM). The final output provides optimal asset allocation and stochastic portfolio simulations.

## Workflow & Architecture

### 1. Data Ingestion & Alternative Data Collection
* **Input Parameters:** The pipeline accepts user-defined stock ticker symbols and a specified historical timeframe.
* **Reddit Scraping:** Scrapes target financial subreddits (e.g., r/wallstreetbets, r/stocks) for any posts and comments mentioning the specified tickers.
* **Relevance Filtering:** Applies a zero-shot classifier to evaluate text context, keeping only posts with a relevance score above `0.7` to eliminate noise.

### 2. Sentiment Analysis (LLM)
* **Llama 3 8B (LoRA):** Utilizes a custom fine-tuned Llama 3 8B model (adapted via LoRA) to analyze the relevance-filtered text.
* **Classification:** Outputs a discrete `Positive`, `Neutral`, or `Negative` sentiment score for each post and comment.
* **Feature Aggregation:** Daily sentiment signals are aggregated per stock and merged with fundamental market data sourced via `yfinance`.

### 3. Quantitative Modeling & Regime Detection
* **GMM-HMM Regime Detection:** Fits a Gaussian Mixture Model Hidden Markov Model (GMM-HMM) to classify market conditions into distinct regimes (e.g., low, medium, and high volatility). This step extracts the probability return distributions within each regime, the transition matrices from one regime to the next, and the stock's current regime.
* **EGARCH Volatility Clusters:** Employs an Exponential Generalized Autoregressive Conditional Heteroskedasticity (EGARCH) model to account for asymmetric volatility clustering—specifically how negative market shocks impact variance more heavily than positive ones. The model outputs the asset's **conditional variance**.

### 4. Predictive Deep Learning
* **LSTM Forecasting:** Feeds the extracted quantitative features (EGARCH conditional variance, current GMM-HMM regime, aggregated sentiment scores, and historical financial data) into a Long Short-Term Memory (LSTM) neural network. The LSTM captures sequential dependencies to predict future market volatility.

### 5. Portfolio Optimization & Simulation
* **Markowitz Optimization:** Uses the LSTM-predicted volatilities and expected returns to compute optimal portfolio weightings under Markowitz Modern Portfolio Theory.
* **Efficient Frontier:** Visualizes the risk-return landscape by plotting the Efficient Frontier.
* **Max Sharpe Ratio:** Automatically identifies and extracts the weightings that yield the highest Sharpe ratio.
* **Stochastic Monte Carlo Simulation:** Runs 500 simulated wealth paths over a 1-year horizon (252 trading days) using the optimized portfolio weights. The simulation samples daily from the probability distributions associated with each stock's current regime and utilizes the calculated transition matrices to dynamically model regime shifts day over day.

## Tech Stack
* **Deep Learning & NLP:** Llama 3 8B, LoRA, Hugging Face Transformers, TensorFlow/Keras (LSTM)
* **Quantitative & Time-Series Models:** `hmmlearn` (GMM-HMM), `arch` (EGARCH)
* **Data Engineering:** `yfinance`, `praw` (Reddit API), Pandas, NumPy
* **Optimization & Visualization:** SciPy, Matplotlib, Seaborn

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
