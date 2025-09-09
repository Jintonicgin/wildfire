import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
import math

# 시각화 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --- Helper Functions ---
def classify_speed(speed: float, thresholds=(0.014, 0.11)) -> int:
    low, high = thresholds
    if speed <= low: return 0
    if speed <= high: return 1
    return 2

def convert_degree_to_direction(deg: float) -> int:
    if deg is None or pd.isna(deg) or deg == -999: return 0
    return int(math.floor(((float(deg) + 22.5) % 360) / 45))

def plot_regression_results(y_true, y_pred, title, output_path):
    """회귀 모델의 실제값 vs 예측값 산점도를 그리고 파일로 저장합니다."""
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.scatter(y_true, y_pred, alpha=0.5, label='예측값')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='완벽한 예측')
    ax.set_xlabel("실제값 (Log 변환)", fontsize=12)
    ax.set_ylabel("예측값 (Log 변환)", fontsize=12)
    ax.set_title(title, fontsize=15, pad=15)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² Score: {r2:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ '{title}' 그래프가 저장되었습니다: {output_path}")

def plot_confusion_matrix(y_true, y_pred, title, labels, output_path):
    """분류 모델의 혼동 행렬을 그리고 파일로 저장합니다."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    ax.set_xlabel('예측된 클래스', fontsize=12)
    ax.set_ylabel('실제 클래스', fontsize=12)
    ax.set_title(title, fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ '{title}' 그래프가 저장되었습니다: {output_path}")

def visualize_predictions():
    """학습된 모델을 로드하여 예측을 수행하고, 실제값과 비교하여 시각화합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 모델 및 데이터 로드
    try:
        area_model = joblib.load(os.path.join(script_dir, "area_regressor_model_v3_tuned.joblib"))
        speed_model = joblib.load(os.path.join(script_dir, "speed_classifier_model_v2_tuned_cw.joblib"))
        direction_model = joblib.load(os.path.join(script_dir, "direction_classifier_model_v2_tuned_cw.joblib"))
        area_scaler = joblib.load(os.path.join(script_dir, "area_model_scaler_v3_tuned.joblib"))
        speed_scaler = joblib.load(os.path.join(script_dir, "speed_scaler_v2_tuned_cw.joblib"))
        direction_scaler = joblib.load(os.path.join(script_dir, "direction_scaler_v2_tuned_cw.joblib"))
        with open(os.path.join(script_dir, "area_model_columns_v3_tuned.json")) as f:
            area_cols = json.load(f)
        with open(os.path.join(script_dir, "speed_model_columns_v2_tuned_cw.json")) as f:
            speed_cols = json.load(f)
        with open(os.path.join(script_dir, "direction_model_columns_v2_tuned_cw.json")) as f:
            direction_cols = json.load(f)
        df = pd.read_csv(os.path.join(script_dir, "final_merged_feature_engineered.csv"))
        df.columns = [col.lower() for col in df.columns]

    except FileNotFoundError as e:
        print(f"❌ Error: 필요한 파일(.joblib, .json, .csv)을 찾을 수 없습니다. ({e.filename})")
        return

    # 2. 데이터 준비
    df.dropna(subset=["fire_area", "fire_duration_hours", "wd10m_0h"], inplace=True)
    df = df[df["fire_duration_hours"] > 0].copy()
    df_filtered = df[(df['fire_area'] > 0) & (df['fire_area'] < df['fire_area'].quantile(0.99))].copy()

    y_area_true = np.log1p(df_filtered['fire_area'])
    df_filtered['spread_speed_class'] = (df_filtered["fire_area"] / df_filtered["fire_duration_hours"]).apply(classify_speed)
    y_speed_true = df_filtered['spread_speed_class']
    df_filtered['spread_direction_class'] = df_filtered["wd10m_0h"].apply(convert_degree_to_direction)
    y_direction_true = df_filtered['spread_direction_class']

    # 3. 예측 수행
    X_area = df_filtered[area_cols].fillna(0)
    X_area_scaled = area_scaler.transform(X_area)
    y_area_pred = area_model.predict(X_area_scaled)

    X_speed = df_filtered[speed_cols].fillna(0)
    X_speed_scaled = speed_scaler.transform(X_speed)
    y_speed_pred = speed_model.predict(X_speed_scaled)

    X_direction = df_filtered[direction_cols].fillna(0)
    X_direction_scaled = direction_scaler.transform(X_direction)
    y_direction_pred = direction_model.predict(X_direction_scaled)

    # 4. 각 그래프를 별도의 파일로 시각화 및 저장
    print("\n--- 실제값 vs. 예측값 비교 시각화 시작 ---")
    plot_regression_results(y_area_true, y_area_pred, '피해 면적 예측 결과', os.path.join(script_dir, "area_prediction_vs_actual.png"))
    plot_confusion_matrix(y_speed_true, y_speed_pred, '확산 속도 예측 결과', ['느림', '보통', '빠름'], os.path.join(script_dir, "speed_prediction_vs_actual.png"))
    plot_confusion_matrix(y_direction_true, y_direction_pred, '확산 방향 예측 결과', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], os.path.join(script_dir, "direction_prediction_vs_actual.png"))
    print("\n🎉 모든 시각화 파일 생성이 완료되었습니다.")

if __name__ == "__main__":
    visualize_predictions()