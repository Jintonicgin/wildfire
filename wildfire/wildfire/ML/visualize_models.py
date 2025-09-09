import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_model_and_columns(model_path: str, columns_path: str):
    """
    저장된 모델과 피처 컬럼 목록을 로드합니다.
    Args:
        model_path (str): .joblib 모델 파일 경로.
        columns_path (str): .json 피처 컬럼 목록 파일 경로.
    Returns:
        tuple: (로드된 모델, 피처 컬럼 목록).
    """
    try:
        model = joblib.load(model_path)
        with open(columns_path, 'r') as f:
            columns = json.load(f)
        print(f"모델 로드 성공: {model_path}")
        return model, columns
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e.filename}")
        return None, None
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

def plot_feature_importance(model, feature_names: list, title: str = "Feature Importance", save_path: str = None, show_plot: bool = True):
    """
    모델의 피처 중요도를 시각화합니다.
    Args:
        model: feature_importances_ 속성을 가진 모델 (예: RandomForestClassifier/Regressor).
        feature_names (list): 피처 이름 목록.
        title (str): 플롯 제목.
        save_path (str, optional): 플롯을 저장할 경로. 지정하지 않으면 저장하지 않습니다.
        show_plot (bool, optional): 플롯을 화면에 표시할지 여부. 기본값은 True.
    """
    if not hasattr(model, 'feature_importances_'):
        print("오류: 이 모델은 feature_importances_ 속성을 가지고 있지 않습니다.")
        return

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, max(6, len(feature_names) * 0.3)))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20)) # 상위 20개 피처만 표시
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs. Predicted", save_path: str = None, show_plot: bool = True):
    """
    회귀 모델의 실제 값과 예측 값을 산점도로 시각화합니다.
    Args:
        y_true (np.ndarray): 실제 값.
        y_pred (np.ndarray): 예측 값.
        title (str): 플롯 제목.
        save_path (str, optional): 플롯을 저장할 경로.
        show_plot (bool, optional): 플롯을 화면에 표시할지 여부.
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2) # y=x 라인
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, title: str = "Confusion Matrix", save_path: str = None, show_plot: bool = True):
    """
    분류 모델의 혼동 행렬을 시각화합니다.
    Args:
        y_true (np.ndarray): 실제 클래스.
        y_pred (np.ndarray): 예측 클래스.
        classes (list): 클래스 레이블 목록.
        title (str): 플롯 제목.
        save_path (str, optional): 플롯을 저장할 경로.
        show_plot (bool, optional): 플롯을 화면에 표시할지 여부.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

if __name__ == '__main__':
    print("이 모듈은 직접 실행 시 예시 데이터를 사용하여 시각화를 수행하지 않습니다.")
    print("실제 모델에 적용하려면, 다른 스크립트에서 이 모듈을 임포트하여 사용하세요.")
    print("예시 사용법:")
    print("  from wildfire.ML.visualize_models import load_model_and_columns, plot_feature_importance, plot_actual_vs_predicted, plot_confusion_matrix")
    print("  # 모델 로드")
    print("  model, features = load_model_and_columns('path/to/your_model.joblib', 'path/to/your_columns.json')")
    print("  if model and features:")
    print("      # 피처 중요도 시각화")
    print("      plot_feature_importance(model, features, title='My Model Feature Importance', save_path='./feature_importance.png')")
    print("      # 회귀 모델의 경우 (예시 데이터)")
    print("      # plot_actual_vs_predicted(np.array([1,2,3]), np.array([1.1,1.9,3.2]), title='Area Model Performance')")
    print("      # 분류 모델의 경우 (예시 데이터)")
    print("      # plot_confusion_matrix(np.array([0,1,0,2]), np.array([0,1,1,2]), classes=[0,1,2], title='Speed Model Confusion Matrix')")
