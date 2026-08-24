# FinAdvisor — Investment Risk Prediction Platform

FinAdvisor is a premium Flask dashboard for analysing investment datasets, predicting risk categories, and translating results into clear portfolio guidance.

## Features

- Automatic CSV/XLSX schema inspection, data profiling, cleaning, outlier capping, and missing-value handling.
- Automatic risk-target detection. If a risk label does not exist, the pipeline infers Low/Medium/High labels from volatility or return metrics.
- Model comparison (Random Forest and Gradient Boosting), holdout accuracy, weighted F1, and cross-validation.
- Saved model, metadata, and automatically generated visual analytics.
- Responsive, Apple-inspired dashboard with an interactive prediction panel.

## Dataset assumptions

The included mutual-funds dataset contains 814 rows and 20 fields. `risk_level` was detected as the supervised target. Missing 3-year and 5-year returns are median-imputed; duplicate records are removed and numerical extremes are capped at the 1st/99th percentiles. Identifier-like fields with near-unique values are excluded from training to improve generalisation.

## Install and run

```bash
cd FinAdvisor
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Replace the dataset

Put one CSV, XLSX, or XLS file in `data/`, remove `model/saved_model.pkl` and `model/metadata.json`, then run `python app.py`. The project will retrain and regenerate visualisations.

## Project layout

`app.py` serves the dashboard; `preprocessing/` discovers and cleans schema; `model/` trains and predicts; `static/images/` stores generated plots.

## Screenshots

Add dashboard screenshots here for a portfolio case study.

## Future improvements

Add authenticated portfolios, scheduled market data refreshes, explainability panels, and personalised risk questionnaires.
