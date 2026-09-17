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
```

---

## Repository Structure

```text
dl-timesnet-xauusd-research/
│
├── app/
│   └── Public research demonstration
│
├── configs/
│   └── Configuration templates
│
├── data/
│   └── Data schema and documentation
│
├── docs/
│   └── Methodology and research notes
│
├── evaluation/
│   └── Evaluation and analysis components
│
├── figures/
│   └── Research figures and visual outputs
│
├── models/
│   └── Model implementations and interfaces
│
├── src/
│   └── Core research utilities
│
├── training/
│   └── Training and experiment components
│
├── CITATION.cff
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Public Research Release

This repository is a public research-oriented implementation.

For data licensing, privacy, reproducibility and research-security reasons, the following components are not included:

- Full proprietary dataset
- Trained production models
- Broker connectivity
- Live execution infrastructure
- Private execution configurations

The public repository therefore focuses on the research methodology, experiment structure and reproducible components that can be shared openly.

---

## Reproducibility

The repository is organized so that the publicly available research components can be inspected independently of any live trading infrastructure.

Configuration files, methodology notes and evaluation components are provided to make the experimental design transparent.

Where proprietary inputs are required, the repository documents the expected data schema rather than distributing the original data.

---

## Research Context

This project is part of an ongoing research programme at the intersection of:

- Financial Econometrics
- Financial Machine Learning
- High-Frequency Financial Time Series
- Deep Sequence Modeling
- Volatility and Regime Modeling
- Risk-Aware Forecasting

The broader objective is to study robust machine-learning methodologies for noisy and non-stationary financial data rather than to provide a commercial trading product.

---

## Citation

If you use this repository or the associated research methodology, please cite:

**Akhlaghi, M. (2026).  
Deep Sequence Architectures in High-Frequency Finance: Benchmarking TimesNet and Classical Machine Learning Across Volatility Regimes. SSRN.**

---

## Author

**Mohammadreza Akhlaghi**

Financial Machine Learning & Econometrics Researcher

- GitHub: https://github.com/hezarah
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7034100
- LinkedIn: Mohammadreza Akhlaghi
