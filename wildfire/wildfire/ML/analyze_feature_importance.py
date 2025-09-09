import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import os
import numpy as np

# 시각화 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_feature_importance(ax, model, columns, title):
    """주어진 모델의 피처 중요도를 바 차트로 그립니다."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # 상위 20개 피처만 선택

    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([columns[i] for i in indices])
    ax.set_xlabel('피처 중요도')
    ax.set_title(title, fontsize=15)

def analyze_model_importance():
    """저장된 분류 모델들을 로드하여 피처 중요도를 분석하고 시각화합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 모델 및 컬럼 정보 로드
    try:
        speed_model = joblib.load(os.path.join(script_dir, "speed_classifier_model_v2_tuned_cw.joblib"))
        direction_model = joblib.load(os.path.join(script_dir, "direction_classifier_model_v2_tuned_cw.joblib"))
        
        with open(os.path.join(script_dir, "speed_model_columns_v2_tuned_cw.json")) as f:
            speed_cols = json.load(f)
        with open(os.path.join(script_dir, "direction_model_columns_v2_tuned_cw.json")) as f:
            direction_cols = json.load(f)
            
    except FileNotFoundError as e:
        print(f"❌ Error: 모델 또는 컬럼 파일을 찾을 수 없습니다. ({e.filename})\n먼저 모든 학습 스크립트를 실행하여 모델 파일을 생성해주세요.")
        return

    # 2. 시각화 Figure 생성
    fig, axes = plt.subplots(2, 1, figsize=(12, 16), constrained_layout=True)
    fig.suptitle('분류 모델 피처 중요도 분석', fontsize=20, weight='bold')

    # 3. 각 모델의 피처 중요도 시각화
    plot_feature_importance(axes[0], speed_model, speed_cols, '확산 속도 예측 모델')
    plot_feature_importance(axes[1], direction_model, direction_cols, '확산 방향 예측 모델')

    # 4. 이미지 파일로 저장
    output_path = os.path.join(script_dir, "feature_importance_analysis.png")
    try:
        plt.savefig(output_path)
        print(f"\n✅ 피처 중요도 분석 그래프가 성공적으로 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"\n❌ 그래프 저장 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    analyze_model_importance()
