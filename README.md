# AI Job Salary Prediction — MLOps Pipeline

A regression pipeline predicting AI/ML job salaries (`salary_usd`) from job posting
attributes, with MLflow experiment tracking, Optuna hyperparameter tuning, a Prefect
orchestration flow, and a Streamlit prediction app.

## Live demo
```bash
pip install -r requirements.txt
streamlit run app.py
```
`model.pkl` is already trained and committed, so the app works immediately —
no training step required first.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — loads `model.pkl` and predicts salary from user input |
| `job_orchestration.py` | Prefect flow: load data → engineer features → train → evaluate → save `model.pkl` |
| `job_pipeline_tracking.py` | Baseline RandomForestRegressor + 50-trial Optuna sweep, logged to MLflow |
| `job_pipeline_hpt.py` | Multi-model sweep (RandomForest / GradientBoosting / KNN), 20 Optuna trials each, nested MLflow runs |
| `ai_job_dataset.csv` | Training data (15,000 rows) |
| `model.pkl` | Trained pipeline (Preprocessor + RandomForestRegressor), Test R² ≈ 0.88 |
| `mlflow.db` | SQLite MLflow backend store |
| `requirements.txt` | Python dependencies |

## Feature engineering

Two engineered features not present in the raw CSV:
- `num_skills` — count of comma-separated entries in `required_skills`
- `days_to_deadline` — `application_deadline - posting_date`, in days

**Numeric features:** `remote_ratio`, `years_experience`, `job_description_length`, `benefits_score`, `num_skills`, `days_to_deadline`
**Categorical features (one-hot encoded):** `job_title`, `experience_level`, `employment_type`, `company_location`, `company_size`, `employee_residence`, `education_required`, `industry`

`job_id`, `company_name`, `salary_currency`, `required_skills`, `posting_date`,
`application_deadline` are dropped after being consumed for feature engineering.

A `ColumnTransformer` (`StandardScaler`/`MinMaxScaler` + `OneHotEncoder`) handles the
mixed numeric/categorical inputs.

## How to run

```bash
pip install -r requirements.txt

# App works immediately — model.pkl is already trained
streamlit run app.py

# To retrain (overwrites model.pkl):
python job_orchestration.py

# To schedule daily retraining via Prefect instead of running once:
python job_orchestration.py --serve

# Optional exploratory runs — log to mlflow.db (each takes several minutes+):
python job_pipeline_tracking.py
python job_pipeline_hpt.py

# Browse MLflow experiment results:
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Notes
- Regression metrics (RMSE, MAE, R²) used throughout — no classification accuracy.
- `KFold` used for cross-validation (no class labels to stratify on in regression).
- `job_orchestration.py` defaults to training once immediately; pass `--serve` for
  the original daily-cron scheduled behavior.
- Both Optuna/MLflow scripts explicitly pin `mlflow.set_tracking_uri("sqlite:///mlflow.db")`
  before any MLflow calls, so runs log to the correct experiment in `mlflow.db`
  rather than falling back to a local `./mlruns` store.
- Predictions are always in **USD** — the model is trained solely on the `salary_usd`
  column; `salary_currency` in the raw data is unused.
