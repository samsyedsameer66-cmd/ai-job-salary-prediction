import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

# Point MLflow at the shared SQLite backend store BEFORE anything else touches
# mlflow (set_experiment, the Optuna MLflowCallback, autolog, etc). If this isn't
# set first, MLflow falls back to a local ./mlruns file store and/or creates a
# stray "no-name-..." experiment instead of the one named below.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Name of the experiment
mlflow.set_experiment("AI_JOB_Salary_Pipeline")

DATA_PATH = "ai_job_dataset.csv"

# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------
data = pd.read_csv(DATA_PATH)

# Data Cleaning
data = data.drop_duplicates()

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
data["posting_date"] = pd.to_datetime(data["posting_date"])
data["application_deadline"] = pd.to_datetime(data["application_deadline"])
data["days_to_deadline"] = (data["application_deadline"] - data["posting_date"]).dt.days
data["num_skills"] = data["required_skills"].apply(lambda s: len(str(s).split(",")))

NUMERIC_FEATURES = [
    "remote_ratio", "years_experience", "job_description_length",
    "benefits_score", "num_skills", "days_to_deadline"
]
CATEGORICAL_FEATURES = [
    "job_title", "experience_level", "employment_type", "company_location",
    "company_size", "employee_residence", "education_required", "industry"
]
TARGET = "salary_usd"

X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = data[TARGET]

# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------------------------
# Define Pipeline
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

pipeline_1 = Pipeline(
    [
        ("Preprocessor", preprocessor),
        ("Model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ]
)


# ---------------------------------------------------------------------------
# Define Objective
# ---------------------------------------------------------------------------
def objective(trial):
    # Hyperparameters suggested by Optuna
    n_estimators = trial.suggest_int("Model__n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("Model__max_depth", 5, 40)
    min_samples_split = trial.suggest_int("Model__min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("Model__min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("Model__max_features", ["sqrt", "log2", None])

    # Set pipeline params for this trial
    pipeline_1.set_params(
        Model__n_estimators=n_estimators,
        Model__max_depth=max_depth,
        Model__min_samples_split=min_samples_split,
        Model__min_samples_leaf=min_samples_leaf,
        Model__max_features=max_features,
    )

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(
        pipeline_1, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf
    ).mean()

    return cv_score  # Optuna will maximize (i.e. minimize RMSE)


mlflow_callback = MLflowCallback(
    tracking_uri="sqlite:///mlflow.db",
    metric_name="cv_neg_rmse"
)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])

best_params = study.best_trial.params
print("Best hyperparameters:", best_params)
print("Best CV neg-RMSE:", study.best_trial.value)

# Autolog final sklearn model
mlflow.sklearn.autolog()

# ---------------------------------------------------------------------------
# Training with Best parameters
# ---------------------------------------------------------------------------
pipeline_1.set_params(
    Model__n_estimators=best_params["Model__n_estimators"],
    Model__max_depth=best_params["Model__max_depth"],
    Model__min_samples_split=best_params["Model__min_samples_split"],
    Model__min_samples_leaf=best_params["Model__min_samples_leaf"],
    Model__max_features=best_params["Model__max_features"],
)
with mlflow.start_run(run_name="best_model_final_fit"):
    pipeline_1.fit(X_train, y_train)

    y_pred_train = pipeline_1.predict(X_train)
    train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
    train_r2 = r2_score(y_train, y_pred_train)
    print("Training RMSE:", train_rmse, "| Training R2:", train_r2)

    # -----------------------------------------------------------------------
    # Testing the model
    # -----------------------------------------------------------------------
    y_pred_test = pipeline_1.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    print("Testing RMSE:", test_rmse, "| Testing MAE:", test_mae, "| Testing R2:", test_r2)

    # Log metrics to MLflow (autolog already captured params/the model itself
    # from .fit(), this just adds the test-set metrics on the same run)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_r2", test_r2)
