# Deep Sequence Modeling for High-Frequency Financial Time Series

Research framework for studying deep sequence architectures on high-frequency financial time series, using 5-minute XAUUSD data as the empirical setting.

The repository accompanies the research paper:

**Deep Sequence Architectures in High-Frequency Finance: Benchmarking TimesNet and Classical Machine Learning Across Volatility Regimes**

[SSRN Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7034100)

---

## Research Objective

This project investigates how modern deep sequence architectures model noisy, non-stationary financial time series and how their predictive behaviour changes across different volatility and market regimes.

The empirical framework compares deep sequence models with classical machine-learning baselines under chronological, out-of-sample evaluation.

The main research questions are:

1. How do deep sequence architectures perform relative to classical machine-learning baselines on high-frequency financial data?
2. How does model performance change across different volatility and market regimes?
3. How can leakage-free labeling and chronological evaluation improve the validity of financial forecasting experiments?
4. To what extent do predictive metrics translate into economically meaningful risk-adjusted outcomes?

---

## Dataset

The empirical study uses 5-minute XAUUSD candlestick data covering 2013–2025.

The research pipeline is designed around:

- Open, High, Low and Close price data
- Derived volatility measures
- Technical and statistical features
- Sequential windows of historical observations
- Chronological train/validation/test evaluation

The full research dataset is not included in this public repository.

---

## Models

The research framework considers multiple model families, including:

### Deep Sequence Models

- TimesNet
- PatchTST
- LSTM
- GRU
- TCN

### Classical Machine Learning Baselines

- Random Forest
- XGBoost
- MLP

The purpose of the comparison is methodological rather than to present a deployable trading system.

---

## Evaluation Framework

Financial time-series experiments are evaluated using chronological and walk-forward principles to reduce look-ahead bias and data leakage.

The evaluation framework considers:

- Out-of-sample predictive performance
- Accuracy
- F1 score
- Matthews Correlation Coefficient (MCC)
- Regime-dependent performance
- Risk-aware trading simulation
- Maximum drawdown
- Risk-adjusted cumulative performance

The methodology is designed to distinguish predictive performance from economic utility.

---

## Research Methodology

The overall workflow is:

```text
Historical Market Data
        │
        ▼
Data Preparation
        │
        ▼
Feature Construction
        │
        ▼
Chronological Windowing
        │
        ▼
Causal / Risk-Aware Labeling
        │
        ▼
Deep Sequence Modeling
        │
        ▼
Classical ML Baselines
        │
        ▼
Out-of-Sample Evaluation
        │
        ▼
Regime-Based Analysis
        │
        ▼
Economic / Risk Evaluation
