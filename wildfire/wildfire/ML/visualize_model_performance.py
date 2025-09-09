import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 시각화 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_regression_performance(ax, data):
    """회귀 모델의 성능을 시각화합니다."""
    r2 = data.get('r2_score', 0)
    rmse = data.get('cv_rmse', 0)
    
    # R-squared 점수 시각화 (바 차트)
    ax.bar(["R² Score"], [r2], color='#2ca02c', width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² Score (설명력)")
    ax.set_title("면적 예측 모델 성능 (회귀)", fontsize=15, pad=20)
    
    # R²와 RMSE 값을 텍스트로 표시
    text_str = f"R² Score: {r2:.4f}\nCV RMSE: {rmse:.4f}"
    ax.text(0, r2 + 0.05, f'{r2:.4f}', ha='center', va='bottom', fontsize=12, color='#2ca02c')
    ax.text(0.5, 0.5, text_str, transform=ax.transAxes, ha='right', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

def plot_classification_performance(ax, speed_data, dir_data):
    """두 분류 모델의 주요 성능 지표를 비교하여 시각화합니다."""
    models = ['속도 예측 모델', '방향 예측 모델']
    
    # 'weighted avg'가 없는 경우를 대비하여 안전하게 값 추출
    speed_metrics = speed_data.get('weighted avg', {})
    dir_metrics = dir_data.get('weighted avg', {})

    precision = [speed_metrics.get('precision', 0), dir_metrics.get('precision', 0)]
    recall = [speed_metrics.get('recall', 0), dir_metrics.get('recall', 0)]
    f1_score = [speed_metrics.get('f1-score', 0), dir_metrics.get('f1-score', 0)]
    
    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    rects1 = ax.bar(x - width, precision, width, label='정밀도 (Precision)', color='#1f77b4')
    rects2 = ax.bar(x, recall, width, label='재현율 (Recall)', color='#ff7f0e')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#2ca02c')

    ax.set_ylabel('점수')
    ax.set_title('속도 & 방향 예측 모델 성능 비교 (분류)', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    # 바 위에 값 표시
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

def visualize_model_performance():
    """저장된 성능 지표를 로드하여 종합적으로 시각화합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 성능 데이터 로드
    try:
        with open(os.path.join(script_dir, "area_model_performance.json")) as f:
            area_perf = json.load(f)
        with open(os.path.join(script_dir, "speed_model_performance.json")) as f:
            speed_perf = json.load(f)
        with open(os.path.join(script_dir, "direction_model_performance.json")) as f:
            direction_perf = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: 성능 지표 파일을 찾을 수 없습니다. ({e.filename})\n먼저 모든 학습 스크립트를 실행하여 성능 파일을 생성해주세요.")
        return

    # 2. 시각화 Figure 생성
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), constrained_layout=True)
    fig.suptitle('모델 성능 종합 대시보드', fontsize=20, weight='bold')

    # 3. 각 subplot에 시각화 함수 호출
    plot_regression_performance(axes[0], area_perf)
    plot_classification_performance(axes[1], speed_perf, direction_perf)

    # 4. 이미지 파일로 저장
    output_path = os.path.join(script_dir, "model_performance_comparison.png")
    try:
        plt.savefig(output_path)
        print(f"\n✅ 모델 성능 비교 대시보드가 성공적으로 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"\n❌ 대시보드 저장 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    visualize_model_performance()
