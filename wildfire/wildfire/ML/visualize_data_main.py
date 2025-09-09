import pandas as pd
import os
from wildfire.ML.visualize_distributions import plot_distributions
from wildfire.ML.train_v3_model import engineer_features
from wildfire.ML.visualize_models import load_model_and_columns # 모델 로드 함수 임포트

# 현재 스크립트의 디렉토리 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 파일 경로
data_path = os.path.join(script_dir, 'final_merged_feature_engineered.csv')

# 플롯을 저장할 기본 디렉토리
base_save_directory = os.path.join(script_dir, 'distribution_plots_important_features')

# 모델 파일 경로 설정
MODEL_PATHS = {
    "area": {
        "model": os.path.join(script_dir, "area_regressor_model_v3_tuned.joblib"),
        "cols": os.path.join(script_dir, "area_model_columns_v3_tuned.json")
    },
    "speed": {
        "model": os.path.join(script_dir, "speed_classifier_model_v2_tuned_cw.joblib"),
        "cols": os.path.join(script_dir, "speed_model_columns_v2_tuned_cw.json")
    },
    "direction": {
        "model": os.path.join(script_dir, "direction_classifier_model_v2_tuned_cw.joblib"),
        "cols": os.path.join(script_dir, "direction_model_columns_v2_tuned_cw.json")
    }
}

def main():
    print("데이터 분포 시각화 스크립트 시작...")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv(data_path)
        print(f"데이터 로드 성공: {data_path} (총 {len(df)} 행, {len(df.columns)} 열)")
    except FileNotFoundError:
        print(f"오류: 데이터 파일 '{data_path}'을(를) 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return

    # 2. 파생 변수 생성
    print("\n파생 변수 생성 중...")
    engineered_features_list = engineer_features(df) # df가 in-place로 수정됨
    print(f"총 {len(engineered_features_list)}개의 새로운 파생 변수 생성 완료.")
    
    # 3. 각 모델별 중요한 변수 분포 시각화
    for model_name, paths in MODEL_PATHS.items():
        print(f"\n--- {model_name.upper()} 모델의 중요한 변수 시각화 시작 ---")
        model, feature_names = load_model_and_columns(paths["model"], paths["cols"])

        if model and feature_names:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                # 상위 30개 중요한 피처 선택
                top_n_features = importance_df['Feature'].head(30).tolist()
                
                # 데이터프레임에서 선택된 피처만 필터링
                # df에 없는 피처는 제외 (예: 데이터 누수로 제거된 피처 등)
                features_to_visualize = [f for f in top_n_features if f in df.columns]
                
                if not features_to_visualize:
                    print(f"경고: {model_name.upper()} 모델의 중요한 피처 중 시각화할 수 있는 피처가 없습니다.")
                    continue

                filtered_df = df[features_to_visualize]
                
                # 모델별 저장 디렉토리 생성
                model_save_directory = os.path.join(base_save_directory, f'{model_name}_model_important_features')
                
                print(f"{model_name.upper()} 모델의 상위 {len(features_to_visualize)}개 중요한 피처 분포 시각화 중...")
                plot_distributions(filtered_df, save_dir=model_save_directory, show_plots=False) # 화면에는 표시하지 않고 저장만
                print(f"{model_name.upper()} 모델 시각화 완료. 결과는 {model_save_directory}에 저장되었습니다.")
            else:
                print(f"경고: {model_name.upper()} 모델은 feature_importances_ 속성을 가지고 있지 않습니다. 피처 중요도를 추출할 수 없습니다.")
        else:
            print(f"오류: {model_name.upper()} 모델 또는 피처 목록을 로드할 수 없습니다. 경로를 확인해주세요.")
    
    print("\n모든 모델의 중요한 변수 시각화 스크립트 완료.")

if __name__ == "__main__":
    main()