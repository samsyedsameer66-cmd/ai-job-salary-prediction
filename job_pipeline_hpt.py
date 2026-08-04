import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

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


def make_preprocessor(scaler_type):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    return ColumnTransformer(
        transformers=[
            ("num", scaler, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


# Base pipeline (preprocessor + model get swapped per trial)
pipeline = Pipeline(
    [
        ("Preprocessor", make_preprocessor("standard")),
        ("Model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ]
)


# ---------------------------------------------------------------------------
# Objective functions per algorithm
# ---------------------------------------------------------------------------
def objective_rf(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    pipeline.set_params(Preprocessor=make_preprocessor(scaler_type))
    pipeline.set_params(
        Model=RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            random_state=42,
            n_jobs=-1
        )
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(
        pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf
    ).mean()
    return score


def objective_gb(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    pipeline.set_params(Preprocessor=make_preprocessor(scaler_type))
    pipeline.set_params(
        Model=GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=42
        )
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(
        pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf
    ).mean()
    return score


def objective_knn(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    pipeline.set_params(Preprocessor=make_preprocessor(scaler_type))
    pipeline.set_params(
        Model=KNeighborsRegressor(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 21, 2),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 3)
        )
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(
        pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kf
    ).mean()
    return score


# Map model names to objective functions
objectives = {
    "RandomForest": objective_rf,
    "GradientBoosting": objective_gb,
    "KNN": objective_knn,
}

# Point MLflow at the shared SQLite backend store BEFORE set_experiment / the
# Optuna MLflowCallback run, otherwise runs land in a local ./mlruns store
# and/or a stray auto-named experiment instead of "AI_JOB_PL_RUNS".
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set experiment
mlflow.set_experiment("AI_JOB_PL_RUNS")

results = {}
model_dict = {}
scaler_dict = {}

# Loop through each algorithm
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(
        tracking_uri="sqlite:///mlflow.db",
        metric_name="cv_neg_rmse",      # Primary metric (str)
        mlflow_kwargs={                 # **MLflow start_run() kwargs**
            "nested": True              # Child runs under parent
        }
    )

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Train the final model
    start_fit = time.time()
    study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    print(f"Best CV neg-RMSE for {model_name}: {study.best_value:.4f}")
    best_params = study.best_params
    results[model_name] = {"best_params": best_params, "best_cv_neg_rmse": study.best_value}

    # Fit the pipeline with the best parameters
    scaler = StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler()
    pipeline.set_params(Preprocessor=make_preprocessor(best_params["scaler_type"]))

    if model_name == "RandomForest":
        pipeline.set_params(
            Model=RandomForestRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"],
                bootstrap=best_params["bootstrap"],
                random_state=42,
                n_jobs=-1
            )
        )
    elif model_name == "GradientBoosting":
        pipeline.set_params(
            Model=GradientBoostingRegressor(
                n_estimators=best_params["n_estimators"],
                learning_rate=best_params["learning_rate"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"],
                subsample=best_params["subsample"],
                random_state=42
            )
        )
    elif model_name == "KNN":
        pipeline.set_params(
            Model=KNeighborsRegressor(
                n_neighbors=best_params["n_neighbors"],
                weights=best_params["weights"],
                p=best_params["p"]
            )
        )

    # Train the final model
    pipeline.fit(X_train, y_train)

    # Evaluate on test data
    start_test = time.time()
    y_pred = pipeline.predict(X_test)
    test_time = time.time() - start_test

    y_pred_train = pipeline.predict(X_train)
    train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
    test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Training RMSE: {train_rmse:.2f}, Testing RMSE: {test_rmse:.2f}, Testing R2: {test_r2:.4f}")
    print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.2f}s")

    # Save model manually to track model size
    model_path = f"{model_name}_final_model.pkl"
    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path)
    for i, model in enumerate(objectives.keys()):
        model_dict[model] = i

    for i, scaler_type in enumerate(["standard", "minmax"]):
        scaler_dict[scaler_type] = i

    with mlflow.start_run(run_name=f"{model_name}_final"):
        mlflow.log_metric("model_id", model_dict[model_name])
        mlflow.log_metric("scaler_id", scaler_dict[best_params["scaler_type"]])
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_time", fit_time)
        mlflow.log_metric("test_time", test_time)
        mlflow.log_metric("model_size", model_size)
        mlflow.sklearn.log_model(pipeline, name=f"{model_name}_ai_job_model")
    os.remove(model_path)

    results[model_name].update({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "fit_time": fit_time,
        "test_time": test_time,
        "model_size_bytes": model_size
    })

# Summary
print("\n--- Summary ---")
for model_name, res in results.items():
    print(f"{model_name}: CV negRMSE={res['best_cv_neg_rmse']:.4f}, Test RMSE={res['test_rmse']:.2f}, "
          f"Test R2={res['test_r2']:.4f}, Fit Time={res['fit_time']:.2f}s, "
          f"Model Size={res['model_size_bytes']} bytes")

# Pick overall best model (lowest test RMSE) and save as model.pkl
best_model_name = min(results, key=lambda k: results[k]["test_rmse"])
print(f"\nBest overall model: {best_model_name} (Test RMSE={results[best_model_name]['test_rmse']:.2f})")
