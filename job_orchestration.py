import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from prefect import task, flow

NUMERIC_FEATURES = [
    "remote_ratio", "years_experience", "job_description_length",
    "benefits_score", "num_skills", "days_to_deadline"
]
CATEGORICAL_FEATURES = [
    "job_title", "experience_level", "employment_type", "company_location",
    "company_size", "employee_residence", "education_required", "industry"
]


@task
def load_data(path: str = "ai_job_dataset.csv"):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(path)


@task
def engineer_features(data):
    """
    Derive extra features from raw columns (skill count, days to deadline).
    """
    data = data.copy()
    data["posting_date"] = pd.to_datetime(data["posting_date"])
    data["application_deadline"] = pd.to_datetime(data["application_deadline"])
    data["days_to_deadline"] = (data["application_deadline"] - data["posting_date"]).dt.days
    data["num_skills"] = data["required_skills"].apply(lambda s: len(str(s).split(",")))
    return data


@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y


@task
def split_train_test(X, y, test_size=0.3, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@task
def build_preprocessor():
    """
    Build the column transformer (scaling + one-hot encoding).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


@task
def train_model(X_train, y_train, preprocessor, hyperparameters):
    """
    Training the machine learning model.
    """
    pipeline = Pipeline(
        [
            ("Preprocessor", preprocessor),
            ("Model", RandomForestRegressor(**hyperparameters, random_state=42, n_jobs=-1))
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


@task
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = metrics.mean_squared_error(y_train, y_train_pred) ** 0.5
    test_rmse = metrics.mean_squared_error(y_test, y_test_pred) ** 0.5
    test_r2 = metrics.r2_score(y_test, y_test_pred)

    return train_rmse, test_rmse, test_r2


@task
def save_model(model, path: str = "model.pkl"):
    """
    Persist the trained pipeline so downstream apps (e.g. the Streamlit app) can load it.
    """
    joblib.dump(model, path)
    return path


# Workflow
@flow(name="AI Job Salary Training Flow")
def workflow():
    DATA_PATH = "ai_job_dataset.csv"
    INPUTS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    OUTPUT = "salary_usd"
    HYPERPARAMETERS = {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2}

    # Load data
    raw = load_data(DATA_PATH)

    # Feature engineering
    data = engineer_features(raw)

    # Identify Inputs and Output
    X, y = split_inputs_output(data, INPUTS, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Preprocessor
    preprocessor = build_preprocessor()

    # Build a model
    model = train_model(X_train, y_train, preprocessor, HYPERPARAMETERS)

    # Evaluation
    train_rmse, test_rmse, test_r2 = evaluate_model(model, X_train, y_train, X_test, y_test)

    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)

    # Persist the model for the Streamlit app
    save_model(model, "model.pkl")


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        # Long-running process: registers a deployment and waits for the cron
        # schedule (next 00:00) or a manual trigger. It does NOT train
        # immediately, so model.pkl won't appear until the schedule fires.
        workflow.serve(
            name="ai-job-salary-deployment",
            cron="0 0 * * *"
        )
    else:
        # Default: run the flow once, right now, so model.pkl is produced
        # immediately for the Streamlit app.
        workflow()
