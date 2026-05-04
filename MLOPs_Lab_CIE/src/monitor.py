import pandas as pd
import json

# Load training data
train = pd.read_csv("data/training_data.csv")

train_slope_mean = train["slope_degrees"].mean()
train_rain_mean = train["rainfall_mm"].mean()

# Load logs
logs = []
with open("logs/predictions.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

live_df = pd.DataFrame([l["input"] for l in logs])

live_slope_mean = live_df["slope_degrees"].mean()
live_rain_mean = live_df["rainfall_mm"].mean()

def check(feature, train_mean, live_mean, threshold):
    shift = abs(live_mean - train_mean)
    status = "ALERT" if shift > threshold else "OK"
    return {
        "feature": feature,
        "train_mean": float(train_mean),
        "live_mean": float(live_mean),
        "shift": float(shift),
        "threshold": threshold,
        "status": status
    }

alerts = [
    check("slope_degrees", train_slope_mean, live_slope_mean, 10.67),
    check("rainfall_mm", train_rain_mean, live_rain_mean, 72.11)
]

output = {
    "total_predictions": len(logs),
    "mean_prediction": sum(l["prediction"] for l in logs) / len(logs),
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

with open("results/step4_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Monitoring complete!")