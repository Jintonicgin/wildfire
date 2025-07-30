import pandas as pd
import json
import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import multiprocessing
import os

# predict.py에서 predict_simulation 함수를 직접 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import predict

# --- 설정 ---
SOURCE_DATA_PATH = "gangwon_fire_data_full_merged.csv"

# --- 각 산불 사례를 처리하는 함수 (병렬 처리용) ---
def process_fire_case(row_data):
    lat = row_data['latitude']
    lon = row_data['longitude']
    
    start_dt = datetime.datetime.strptime(row_data['start_datetime'], '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.datetime.strptime(row_data['end_datetime'], '%Y-%m-%d %H:%M:%S')
    
    duration_td = end_dt - start_dt
    duration_hours = max(1, int(duration_td.total_seconds() / 3600)) # 최소 1시간

    actual_area = row_data['damage_area']
    timestamp_iso = start_dt.isoformat()

    # 진행 상황 출력 (각 프로세스에서 출력되므로 순서가 섞일 수 있음)
    print(f"[PID {os.getpid()}] Processing case for Lat={lat:.4f}, Lng={lon:.4f}, Duration={duration_hours}h")

    # predict.py의 predict_simulation 함수를 직접 호출
    input_data = {
        "latitude": lat,
        "longitude": lon,
        "timestamp": timestamp_iso,
        "durationHours": duration_hours
    }
    prediction_result = predict.predict_simulation(input_data)

    if "error" in prediction_result:
        predicted_area = np.nan # 예측 실패 시 NaN 처리
    else:
        predicted_area = prediction_result['final_damage_area']
    
    return actual_area, predicted_area

# --- 메인 실행 로직 ---
def main():
    print(f"1. 데이터 로딩: {SOURCE_DATA_PATH}")
    try:
        df = pd.read_csv(SOURCE_DATA_PATH)
    except FileNotFoundError:
        print(f"오류: 데이터 파일 '{SOURCE_DATA_PATH}'을 찾을 수 없습니다.")
        return

    print("2. 각 산불 사례에 대해 시뮬레이션 예측 병렬 수행 (시간이 오래 걸릴 수 있습니다)")
    
    # CPU 코어 수에 따라 프로세스 풀 생성
    num_processes = multiprocessing.cpu_count()
    print(f"  - {num_processes}개의 프로세스를 사용하여 병렬 처리합니다.")
    
    # DataFrame의 각 행을 딕셔너리로 변환하여 전달
    rows_as_dicts = df.to_dict(orient='records')

    with multiprocessing.Pool(processes=num_processes, initializer=predict._initialize_prediction_environment) as pool:
        results = pool.map(process_fire_case, rows_as_dicts)

    actual_damage_areas = [res[0] for res in results]
    predicted_damage_areas = [res[1] for res in results]

    # NaN 값 제거 (예측 실패 사례 제외)
    valid_indices = ~np.isnan(predicted_damage_areas)
    actual_damage_areas = np.array(actual_damage_areas)[valid_indices]
    predicted_damage_areas = np.array(predicted_damage_areas)[valid_indices]

    if len(actual_damage_areas) == 0:
        print("모든 예측이 실패했거나 유효한 데이터가 없습니다. 성능을 평가할 수 없습니다.")
        return

    print("\n3. 시뮬레이션 성능 평가")
    rmse = np.sqrt(mean_squared_error(actual_damage_areas, predicted_damage_areas))
    r2 = r2_score(actual_damage_areas, predicted_damage_areas)

    print(f"  - 시뮬레이션 RMSE: {rmse:.4f}")
    print(f"  - 시뮬레이션 R^2: {r2:.4f}")

    print("\n4. 결과 시각화")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=actual_damage_areas, y=predicted_damage_areas, alpha=0.6)
    plt.plot([min(actual_damage_areas), max(actual_damage_areas)], 
             [min(actual_damage_areas), max(actual_damage_areas)], 'r--', lw=2)
    plt.xlabel("Actual Damage Area (ha)")
    plt.ylabel("Predicted Damage Area (ha)")
    plt.title("Simulation: Actual vs. Predicted Damage Area")
    plt.grid(True)
    plt.savefig("simulation_validation_scatter.png")
    print("  - 산점도 그래프 저장 완료: simulation_validation_scatter.png")

    # 잔차도
    residuals = actual_damage_areas - predicted_damage_areas
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=predicted_damage_areas, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Damage Area (ha)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Simulation: Residual Plot")
    plt.grid(True)
    plt.savefig("simulation_validation_residuals.png")
    print("  - 잔차도 그래프 저장 완료: simulation_validation_residuals.png")

    print("\n시뮬레이션 검증 완료!")

if __name__ == "__main__":
    # Windows에서는 multiprocessing을 사용할 때 if __name__ == "__main__": 블록이 필수
    multiprocessing.freeze_support()
    main()