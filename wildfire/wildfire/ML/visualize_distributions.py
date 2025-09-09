import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_distributions(df: pd.DataFrame, save_dir: str = None, show_plots: bool = True):
    """
    데이터프레임 내 변수들의 분포를 시각화합니다.
    숫자형 변수는 히스토그램과 KDE 플롯을, 범주형 변수는 막대 그래프를 생성합니다.

    Args:
        df (pd.DataFrame): 시각화할 데이터프레임.
        save_dir (str, optional): 플롯을 저장할 디렉토리 경로. 지정하지 않으면 저장하지 않습니다.
        show_plots (bool, optional): 플롯을 화면에 표시할지 여부. 기본값은 True.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"저장 디렉토리 생성: {save_dir}")

    print("변수 분포 시각화 시작...")

    # 숫자형 변수 시각화
    numerical_cols = df.select_dtypes(include=np.number).columns
    if not numerical_cols.empty:
        print(f"\n숫자형 변수 ({len(numerical_cols)}개) 분포 시각화:")
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{col}_distribution.png'))
            if show_plots:
                plt.show()
            plt.close()
    else:
        print("\n숫자형 변수가 없습니다.")

    # 범주형 변수 시각화 (고유값 수가 적은 경우)
    categorical_cols = df.select_dtypes(include='object').columns
    # 숫자형이지만 고유값 수가 적어 범주형처럼 다룰 수 있는 변수 추가
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique() < 20 and df[col].nunique() > 1: # 고유값 1개는 제외
            categorical_cols = categorical_cols.append(pd.Index([col]))
    
    if not categorical_cols.empty:
        print(f"\n범주형 변수 ({len(categorical_cols)}개) 분포 시각화:")
        for col in categorical_cols:
            # 숫자형으로 인식된 범주형 변수의 경우, 정수형으로 변환하여 플롯
            if df[col].dtype in [np.int64, np.float64]:
                value_counts = df[col].value_counts().sort_index()
            else:
                value_counts = df[col].value_counts()

            plt.figure(figsize=(10, 6))
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{col}_category_distribution.png'))
            if show_plots:
                plt.show()
            plt.close()
    else:
        print("\n범주형 변수가 없습니다.")

    print("\n변수 분포 시각화 완료.")

if __name__ == '__main__':
    # 예시 데이터 생성
    data = {
        'numerical_feature_1': np.random.randn(1000),
        'numerical_feature_2': np.random.rand(1000) * 100,
        'categorical_feature_1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'categorical_feature_2': np.random.randint(0, 5, 1000), # 숫자형이지만 범주형처럼
        'binary_feature': np.random.choice([0, 1], 1000),
        'nan_feature': np.random.choice([1, 2, 3, np.nan], 1000)
    }
    sample_df = pd.DataFrame(data)

    # 사용 예시
    # plot_distributions(sample_df, save_dir='./distribution_plots', show_plots=True)
    print("이 모듈은 직접 실행 시 예시 데이터를 사용하여 시각화를 수행합니다.")
    print("실제 데이터에 적용하려면, 다른 스크립트에서 이 모듈을 임포트하여 사용하세요.")
