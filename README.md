# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave a service.

## What it demonstrates
- Data cleaning and exploratory analysis
- Train/test split with stratification
- Preprocessing numeric and categorical features with `ColumnTransformer`
- Classification with Logistic Regression and Random Forest
- Evaluation using precision, recall, F1-score, ROC-AUC, and confusion matrix
- Saved model pipeline and a Streamlit prediction interface

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The included CSV is a small **synthetic demo dataset** for running the project immediately. It is not real customer data and should not be used to claim real-world performance. For a portfolio version, replace it with a public telecom churn dataset, document its source, and rerun the evaluation.

## Project structure
- `data/churn.csv` — synthetic sample data
- `train.py` — trains, evaluates, and saves the pipeline
- `app.py` — interactive prediction UI
- `artifacts/` — created after training
