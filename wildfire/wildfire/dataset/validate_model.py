import pandas as pd
import datetime
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# validation_prediction_logic.py 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation_prediction_logic import predict_simulation_for_validation

# ✅ 사용하려는 CSV 파일 경로
CSV_FILE_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/gangwon_fire_data_augmented_parallel.csv"

def validate_model():
    print(f"\n모델 검증 시작: {CSV_FILE_PATH} 파일 로드...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"✅ {len(df)}개의 산불 데이터 행을 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: {CSV_FILE_PATH} 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"❌ CSV 파일 읽기 중 오류 발생: {e}")
        return

    actual_damage_areas = []
    predicted_damage_areas = []

    print("\n각 산불 이벤트에 대해 예측 수행 중 (CSV 데이터 기반, API 호출 없음!)...")

    for index, row in df.iterrows():
        try:
            lat = row['lat']
            lon = row['lng']
            actual_damage_area = row['final_damage_area_ha']
            fire_duration_hours = row.get('duration_hours', 1)
            start_datetime_str = row['start_datetime']
        except KeyError as ke:
            print(f"❌ 누락된 컬럼: {ke} (행 {index})")
            continue

        try:
            start_timestamp = datetime.datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"⚠️ start_datetime 파싱 오류: {start_datetime_str} (행 {index}) - 건너뜀")
            continue

        try:
            simulation_hours = int(fire_duration_hours) if pd.notna(fire_duration_hours) else 1
        except Exception:
            print(f"⚠️ duration 변환 오류: {fire_duration_hours} (행 {index}) - 건너뜀")
            continue

        input_json = {
            "latitude": lat,
            "longitude": lon,
            "timestamp": start_timestamp.isoformat(),
            "durationHours": simulation_hours
        }

        try:
            prediction_result = predict_simulation_for_validation(input_json, row)
            predicted_damage_area = prediction_result["final_damage_area"]

            actual_damage_areas.append(actual_damage_area)
            predicted_damage_areas.append(predicted_damage_area)

            if (index + 1) % 10 == 0:
                print(f"✅ {index + 1}/{len(df)} 예측 완료. 실제: {actual_damage_area:.2f}, 예측: {predicted_damage_area:.2f}")

        except Exception as e:
            print(f"❌ 예측 실패 (행 {index}, 위도: {lat}, 경도: {lon}, 시간: {start_datetime_str}): {e}")
            continue

    if len(actual_damage_areas) == 0:
        print("⚠️ 예측에 성공한 데이터가 없어 정확도를 계산할 수 없습니다.")
        return

    # 정확도 지표 계산
    mae = mean_absolute_error(actual_damage_areas, predicted_damage_areas)
    rmse = np.sqrt(mean_squared_error(actual_damage_areas, predicted_damage_areas))
    r2 = r2_score(actual_damage_areas, predicted_damage_areas)

    print("\n--- 모델 검증 결과 ---")
    print(f"총 검증 데이터 수: {len(actual_damage_areas)}건")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    print("----------------------")

if __name__ == "__main__":
    validate_model()