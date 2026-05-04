import os

# ✅ Fix working directory (VERY IMPORTANT)
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("land_stability_score", axis=1)
y = df["land_stability_score"]

# Train-test split (as required)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow setup
mlflow.set_experiment("geosurvey-land-stability-score")

results = []

models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

# Create folders if not present
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Train models
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("priority", "high")

        mlflow.sklearn.log_model(model, name)

        # Save model locally
        joblib.dump(model, f"models/{name}.pkl")

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

# Select best model (based on MAE)
best_model = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "geosurvey-land-stability-score",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "mae",
    "best_metric_value": float(best_model["mae"])
}

# Save results JSON
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Training complete!")