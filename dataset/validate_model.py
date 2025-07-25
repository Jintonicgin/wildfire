import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 설정 ---
SOURCE_DATA_PATH = "gangwon_fire_data_augmented_parallel.csv"
MODEL_PATH = "area_regressor_model_v3.joblib"
SCALER_PATH = "area_model_scaler_v3.joblib"
COLUMNS_PATH = "area_model_columns_v3.json"

# --- 시각화 함수 ---
def plot_results(y_true, y_pred):
    plt.figure(figsize=(18, 8))
    
    # 1. 실제값 vs 예측값 산점도
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Damage Area (ha)")
    plt.ylabel("Predicted Damage Area (ha)")
    plt.title("Actual vs. Predicted Damage Area")
    plt.grid(True)

    # 2. 잔차도
    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Damage Area (ha)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid(True)

    plt.tight_layout()
    output_path = "validation_plots.png"
    plt.savefig(output_path)
    print(f"\n검증 시각화 그래프 저장 완료: {output_path}")

# --- 메인 실행 로직 ---
def main():
    print("1. 모델, 스케일러, 피처 정보 로딩")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLUMNS_PATH, 'r') as f:
        features = json.load(f)

    print("2. 전체 데이터 로딩 및 준비")
    df = pd.read_csv(SOURCE_DATA_PATH)
    
    target = 'estimated_damage_area'
    df.dropna(subset=features + [target], inplace=True)
    df = df[(df[target] > 0) & (df[target] < df[target].quantile(0.99))]

    X = df[features]
    y = np.log1p(df[target])

    # 훈련/테스트 데이터 분할 (train_v2_model.py와 동일한 random_state 사용)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. 테스트 데이터 스케일링 및 예측")
    X_test_scaled = scaler.transform(X_test)
    
    y_pred_log = model.predict(X_test_scaled)

    # 원래 스케일로 복원
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)

    print("4. 모델 성능 재검증")
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print(f"  - 검증 데이터 RMSE: {rmse:.4f}")
    print(f"  - 검증 데이터 R^2: {r2:.4f}")

    # 5. 결과 시각화
    plot_results(y_test_orig, y_pred_orig)

if __name__ == "__main__":
    main()
