# Deep Learning Time-Series Research on XAUUSD (TimesNet)

This repository provides a **research-oriented deep learning framework** for modeling XAUUSD using **5-minute candlestick data (2013–2025)**.

The public code is released as a **Research-only edition**:
- methodology and evaluation protocol are reproducible
- suitable for academic review and PhD applications
- **without** any broker connectivity, live trading, proprietary datasets, or commercial execution rules

## Highlights
- Sequence modeling with TimesNet-style architectures
- Window-based time-series formulation
- Research-grade evaluation (out-of-sample + robustness perspective)
- Figures and analysis consistent with the accompanying Research Report

## What is intentionally excluded (IP protection)
- full dataset
- trained weights
- MetaTrader/MT5 connectivity
- production inference-to-execution pipeline
- exact commercial risk and execution parameters

The public release focuses on methodology and evaluation; the private edition includes live connectivity and commercial execution logic.


## Repository Structure
- `app/` Streamlit research demo (public-safe)
- `src/` lightweight stubs and schema definitions
- `configs/` configuration templates
- `data/` schema only (no proprietary data)
- `figures/` example outputs used in the report
- `docs/` methodology and evaluation notes

## Run (demo)
```bash
pip install -r requirements.txt
streamlit run app/app_public.py

## Disclaimer
This repository is for academic research only and does not provide financial advice or a deployable trading product.

## Research Context
This repository accompanies a broader research report comparing classical ML and deep learning approaches for financial time-series modeling under realistic market constraints.

## Author
MohammadReza Akhlaghi

