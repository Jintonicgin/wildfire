import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 시각화 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' # macOS 사용자
# plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 사용자
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

def visualize_feature_correlations():
    """
    사용자가 직접 지정한 피처들의 상관관계를 계산하고 히트맵으로 시각화합니다.
    """
    # 1. 데이터 로드
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "final_merged_feature_engineered.csv")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {data_path}")
        return

    df.columns = [col.lower() for col in df.columns]

    # 2. 사용자가 지정한 피처 목록 사용 (단축된 이름 버전)
    user_feature_list = [
        'potential_spread_index',
        'fuel_combo',
        'dry_windy_combo',
        'fwi_0h',
        'dc_0h',
        'bui_0h',
        'ws10m_max_24h_past',
        'consecutive_dry_days_start',
        'isi_0h',
        'dmc_0h',
        'slope_max',
        't2m_max_24h_past',
        'rh2m_min_24h_past',
        'elevation_std',
        'dry_days_90d_start',
        'ffmc_0h',
        'ws10m_0h_past',
        'slope_mean',
        'ndvi_before',
        'treecover_pre_fire_5x5'
    ]

    # 데이터프레임에 존재하는 피처만 필터링
    available_features = [f for f in user_feature_list if f in df.columns]
    print(f"분석에 사용될 피처 ({len(available_features)}개):\n{available_features}")
    
    missing_features = set(user_feature_list) - set(available_features)
    if missing_features:
        print(f"\n⚠️ 경고: 다음 피처는 데이터 파일에 존재하지 않습니다: {sorted(list(missing_features))}")

    if len(available_features) < 2:
        print("분석할 피처가 충분하지 않습니다.")
        return

    df_selected = df[available_features].copy()

    # 3. 데이터 정제
    original_rows = len(df_selected)
    for col in df_selected.columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
    df_selected.replace(-999, np.nan, inplace=True)
    df_selected.dropna(inplace=True)
    cleaned_rows = len(df_selected)
    print(f"\n데이터 정제 완료: 결측치 포함 행 {original_rows - cleaned_rows}개 제거. 분석에 {cleaned_rows}개 행 사용.")

    if cleaned_rows < 2:
        print("분석할 데이터가 충분하지 않습니다.")
        return

    # 4. 상관관계 행렬 계산
    corr_matrix = df_selected.corr(method='spearman')

    # 5. 히트맵 시각화
    plt.figure(figsize=(18, 15))
    sns.heatmap(
        corr_matrix, 
        annot=True,      
        cmap='coolwarm', 
        fmt='.2f',       
        linewidths=.5,
        annot_kws={"size": 8}
    )
    
    plt.title(f'사용자 지정 피처 간 상관관계 히트맵 (Spearman, {len(available_features)}개)', size=20, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 6. 이미지 파일로 저장
    output_path = os.path.join(script_dir, "feature_correlation_heatmap_custom.png")
    try:
        plt.savefig(output_path)
        print(f"\n✅ 히트맵이 성공적으로 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"\n❌ 히트맵 저장 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    visualize_feature_correlations()