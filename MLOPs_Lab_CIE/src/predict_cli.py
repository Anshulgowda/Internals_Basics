import argparse
import joblib
import json

parser = argparse.ArgumentParser()

parser.add_argument("--slope_degrees", type=float, required=True)
parser.add_argument("--rainfall_mm", type=float, required=True)
parser.add_argument("--soil_depth_m", type=float, required=True)
parser.add_argument("--vegetation_index", type=float, required=True)

args = parser.parse_args()

model = joblib.load("models/RandomForest.pkl")

features = [[
    args.slope_degrees,
    args.rainfall_mm,
    args.soil_depth_m,
    args.vegetation_index
]]

prediction = model.predict(features)[0]

print(json.dumps({"prediction": float(prediction)}))