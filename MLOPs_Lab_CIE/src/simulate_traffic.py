import requests
import random

url = "http://127.0.0.1:8080/predict"

# 30 normal requests
for _ in range(30):
    data = {
        "slope_degrees": random.uniform(5, 45),
        "rainfall_mm": random.uniform(50, 500),
        "soil_depth_m": random.uniform(0.5, 5),
        "vegetation_index": random.uniform(0.1, 0.9)
    }
    requests.post(url, json=data)

# 20 drifted requests
for _ in range(20):
    data = {
        "slope_degrees": random.uniform(50, 90),
        "rainfall_mm": random.uniform(600, 1000),
        "soil_depth_m": random.uniform(0.5, 5),
        "vegetation_index": random.uniform(0.1, 0.9)
    }
    requests.post(url, json=data)

print("✅ Traffic simulation done!")