import os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import datetime
import json

# Load model
model = joblib.load("models/RandomForest.pkl")

# Create app
app = FastAPI()

class InputData(BaseModel):
    slope_degrees: float = Field(..., ge=5, le=90)
    rainfall_mm: float = Field(..., ge=50, le=1000)
    soil_depth_m: float = Field(..., ge=0.5, le=5)
    vegetation_index: float = Field(..., ge=0.1, le=1)

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForest",
        "version": "1.0"
    }

@app.post("/predict")
def predict(data: InputData):
    features = [[
        data.slope_degrees,
        data.rainfall_mm,
        data.soil_depth_m,
        data.vegetation_index
    ]]

    prediction = model.predict(features)[0]

    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "input": data.dict(),
        "prediction": float(prediction)
    }

    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": float(prediction)}