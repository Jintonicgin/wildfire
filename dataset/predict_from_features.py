import sys
import json
import pandas as pd
import numpy as np
import joblib
import warnings

# --- Custom EnsembleRegressor for unpickling ---
class EnsembleRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)

warnings.filterwarnings("ignore")

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

# --- 모델 및 스케일러 로드 ---
MODELS = {
    "area_model": joblib.load(MODEL_PATH + "area_regressor_model_v2.joblib"),
    "speed_model": joblib.load(MODEL_PATH + "speed_classifier_model.joblib"),
    "direction_model": joblib.load(MODEL_PATH + "direction_classifier_model.joblib"),
    "area_scaler": joblib.load(MODEL_PATH + "area_model_scaler_v2.joblib"),
    "speed_scaler": joblib.load(MODEL_PATH + "speed_model_scaler.joblib"),
}
with open(MODEL_PATH + "area_model_columns_v2.json") as f:
    MODELS["area_cols"] = json.load(f)
with open(MODEL_PATH + "speed_model_columns.json") as f:
    MODELS["speed_cols"] = json.load(f)
with open(MODEL_PATH + "direction_model_columns.json") as f:
    MODELS["direction_cols"] = json.load(f)

# --- 예측 함수 ---
def predict_from_features(features):
    df = pd.DataFrame([features])

    area_input = df[MODELS["area_cols"]].copy().fillna(0)
    speed_input = df[MODELS["speed_cols"]].copy().fillna(0)
    direction_input = df[MODELS["direction_cols"]].copy().fillna(0)

    area_scaled = MODELS["area_scaler"].transform(area_input)
    speed_scaled = MODELS["speed_scaler"].transform(speed_input)

    area_log = MODELS["area_model"].predict(area_scaled)[0]
    area = float(np.expm1(area_log))
    if not np.isfinite(area) or area < 0: area = 0.0

    spread_dist_m = float(np.sqrt(area * 10000 / np.pi))
    speed_cat = int(MODELS["speed_model"].predict(speed_scaled)[0])
    direction_class = str(MODELS["direction_model"].predict(direction_input)[0])

    return {
        "predicted_area": area,
        "predicted_distance_m": spread_dist_m,
        "spread_speed_category": speed_cat,
        "spread_direction": direction_class,
    }

# --- 메인 실행 ---
if __name__ == "__main__":
    try:
        input_json = json.loads(sys.stdin.read())
        result = predict_from_features(input_json)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        import traceback
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))