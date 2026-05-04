# 🌍 GeoSurvey Land Stability Prediction (MLOps Lab CIE)

This project implements an end-to-end **MLOps pipeline** for predicting **land stability scores** using geotechnical data.
It includes model training, experiment tracking, containerization, API serving, prediction logging, and drift monitoring.

---

## 📌 Problem Statement

GeoSurvey provides geotechnical assessment services.
The goal is to predict **land stability score** based on:

* slope_degrees
* rainfall_mm
* soil_depth_m
* vegetation_index

---

## 📁 Project Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │   ├── training_data.csv
    │   └── new_data.csv
    │
    ├── src/
    │   ├── train.py
    │   ├── predict_cli.py
    │   ├── api.py
    │   ├── simulate_traffic.py
    │   └── monitor.py
    │
    ├── models/
    ├── logs/
    ├── results/
    │   ├── step1_s1.json
    │   ├── step2_s3.json
    │   ├── step3_s4.json
    │   └── step4_s5.json
    │
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

---

# ⚙️ Setup Instructions

## 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 2️⃣ Train Models (Task 1)

```
python src/train.py
```

### ✔️ What this does:

* Trains **SVR** and **RandomForest**
* Logs experiments using **MLflow**
* Saves models in `models/`
* Generates:

```
results/step1_s1.json
```

---

# 🧠 Task 1 — Experiment Tracking

* Experiment name: `geosurvey-land-stability-score`
* Metrics:

  * MAE
  * RMSE
* Best model selected based on **lowest MAE**

---

# 🐳 Task 2 — Docker Packaging

## Build Docker Image

```
docker build -t geosurvey-predictor:v1 .
```

## Run Container

```
docker run geosurvey-predictor:v1 \
--slope_degrees 31.5 \
--rainfall_mm 196.7 \
--soil_depth_m 3.3 \
--vegetation_index 0.5
```

### ✔️ Output

```
{"prediction": <value>}
```

### ✔️ Result File

```
results/step2_s3.json
```

---

# 🌐 Task 3 — FastAPI Serving

## Run API

```
python -m uvicorn src.api:app --reload --port 8080
```

---

## Endpoints

### 🔹 Health Check

```
GET /health
```

Response:

```
{
  "status": "running",
  "model": "RandomForest",
  "version": "1.0"
}
```

---

### 🔹 Prediction

```
POST /predict
```

Example input:

```
{
  "slope_degrees": 31.5,
  "rainfall_mm": 196.7,
  "soil_depth_m": 3.3,
  "vegetation_index": 0.5
}
```

---

## Swagger UI

```
http://127.0.0.1:8080/docs
```

---

## Output File

```
results/step3_s4.json
```

---

# 📊 Task 4 — Logging & Monitoring

## 1️⃣ Simulate Traffic

```
python src/simulate_traffic.py
```

* Sends:

  * 30 normal requests
  * 20 drifted requests

---

## 2️⃣ Run Monitoring

```
python src/monitor.py
```

---

## ✔️ Features

* Logs predictions to:

```
logs/predictions.jsonl
```

* Detects drift using:

  * slope_degrees threshold = 10.67
  * rainfall_mm threshold = 72.11

---

## ✔️ Output File

```
results/step4_s5.json
```

---

# 📈 Drift Detection Logic

* Compare **training mean vs live mean**
* If difference exceeds threshold → ALERT

---

# 🧾 Outputs Summary

| Task   | Output File   |
| ------ | ------------- |
| Task 1 | step1_s1.json |
| Task 2 | step2_s3.json |
| Task 3 | step3_s4.json |
| Task 4 | step4_s5.json |

---

# ⚠️ Important Notes

* Do NOT modify CSV data
* All outputs must be **JSON (no screenshots)**
* Ensure API runs on port **8080**
* Repository must be **public**

---

# 🚀 Technologies Used

* Python
* Scikit-learn
* MLflow
* FastAPI
* Docker
* Pandas & NumPy

---

# ✅ Conclusion

This project demonstrates a complete **MLOps pipeline**, including:

* Model training & tracking
* Deployment via Docker
* API-based serving
* Real-time logging
* Drift detection & monitoring

---

# 👨‍💻 Author

**Anshul Gowda**
BMS College of Engineering
MLOps Lab CIE (VII Semester)
