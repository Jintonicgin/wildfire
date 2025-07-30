
import sys
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/"

# --- 모델을 만들 때 사용된 클래스 정의 추가 ---
class EnsembleRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)

# --- v2 모델과 컬럼 로드 ---
try:
    area_model = joblib.load(MODEL_PATH + "area_regressor_model_v2.joblib")
    area_scaler = joblib.load(MODEL_PATH + "area_model_scaler_v2.joblib")
    with open(MODEL_PATH + "area_model_columns_v2.json") as f:
        area_cols_expected = json.load(f)
except Exception as e:
    print(json.dumps({"error": f"모델 또는 스케일러 로드 실패: {e}"}))
    sys.exit(1)

# --- 메인 예측 함수 ---
def predict_from_features(input_json):
    try:
        df_raw = pd.DataFrame([input_json])
        df_raw.columns = [c.lower() for c in df_raw.columns]

        rename_map = {
            "t2m": "T2M", "rh2m": "RH2M", "ws10m": "WS10M", "wd10m": "WD10M",
            "prectotcorr": "PRECTOTCORR", "ffmc": "FFMC", "dmc": "DMC", "dc": "DC",
            "isi": "ISI", "bui": "BUI", "fwi": "FWI", "lat": "lat", "lng": "lng"
        }
        df = df_raw.rename(columns=rename_map)

        duration_hours = df_raw.get("durationhours", [1])[0]
        df["duration_hours"] = duration_hours
        df["total_duration_hours"] = duration_hours

        for col in area_cols_expected:
            if col not in df.columns:
                df[col] = 0.0
        
        final_input = df[area_cols_expected]
        final_scaled = area_scaler.transform(final_input)
        area_log = area_model.predict(final_scaled)[0]
        area = float(np.expm1(area_log))

        # 12시간 이상 예측 시, 피해 면적이 비정상적으로 감소하는 것을 방지하는 보정 로직
        if duration_hours >= 12:
            # 9시간일 때의 예측값을 기준으로 최소값 설정 (경험적 보정)
            df_9hr = final_input.copy()
            df_9hr["duration_hours"] = 9
            df_9hr["total_duration_hours"] = 9
            scaled_9hr = area_scaler.transform(df_9hr)
            area_log_9hr = area_model.predict(scaled_9hr)[0]
            area_9hr = float(np.expm1(area_log_9hr))
            area = max(area, area_9hr) # 12시간 예측이 9시간 예측보다 작으면 9시간 값 사용

        if not np.isfinite(area) or area < 0:
            area = 0.0

        # --- 결과 반환 (JSON 직렬화를 위해 표준 Python 타입으로 변환) ---
        return {
            "simulation_hours": int(duration_hours),
            "final_damage_area": float(area),
            "final_lat": float(df.get("lat", [-999])[0]),
            "final_lon": float(df.get("lng", [-999])[0]),
            "path_trace": [{
                "hour": 1,
                "lat": float(df.get("lat", [-999])[0]),
                "lon": float(df.get("lng", [-999])[0]),
                "hourly_damage_area": float(area),
                "cumulative_damage_area": float(area),
                "wind_direction_deg": float(df.get("WD10M", [-999])[0])
            }]
        }

    except Exception as e:
        import traceback
        return {"error": f"피처 기반 예측 실패: {str(e)}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    try:
        input_data = json.loads(sys.stdin.read())
        result = predict_from_features(input_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"스크립트 실행 실패: {str(e)}", "traceback": traceback.format_exc()}))
