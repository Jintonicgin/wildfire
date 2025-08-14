import pandas as pd
import numpy as np
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# predict.py import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_simulation
from predict import initialize_gee_once


SOURCE_DATA_PATH = "gangwon_fire_data_full_merged.csv"

def run_case(row):
    lat = row['latitude']
    lon = row['longitude']
    start = datetime.datetime.strptime(row['start_datetime'], "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime(row['end_datetime'], "%Y-%m-%d %H:%M:%S")
    duration = max(1, int((end - start).total_seconds() / 3600))

    input_data = {
        "latitude": lat,
        "longitude": lon,
        "timestamp": start.isoformat(),
        "durationHours": duration
    }

    try:
        pred = predict_simulation(input_data)
        pred_area = pred.get("final_damage_area", np.nan)
    except Exception as e:
        pred_area = np.nan

    return (row['damage_area'], pred_area)

def main():
    print(f"📥 Loading data from: {SOURCE_DATA_PATH}")
    df = pd.read_csv(SOURCE_DATA_PATH)
    print(f"📊 Total cases: {len(df)}")

    results = []
    print("🚀 Running simulation predictions...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_case, row) for _, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
            results.append(future.result())

    actual, predicted = zip(*results)
    actual = np.array(actual)
    predicted = np.array(predicted)

    valid = ~np.isnan(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        print("⚠️ No valid predictions.")
        return

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R²: {r2:.4f}")

    # 산점도
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=actual, y=predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel("Actual Area (ha)")
    plt.ylabel("Predicted Area (ha)")
    plt.title("Actual vs Predicted Area")
    plt.grid(True)
    plt.savefig("fast_validation_scatter.png")

    # 잔차도
    residuals = actual - predicted
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Area")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.grid(True)
    plt.savefig("fast_validation_residuals.png")

    print("📈 시각화 완료. 파일 저장됨.")

if __name__ == "__main__":
    initialize_gee_once()
    main()