import pandas as pd
import os
import numpy as np

def check_all_columns():
    """CSV 파일의 모든 숫자형 컬럼 목록을 출력합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "final_merged_feature_engineered.csv")
    
    try:
        # DtypeWarning을 피하기 위해 low_memory=False 옵션 추가
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"❌ Error: 파일을 찾을 수 없습니다: {file_path}")
        return

    # 대소문자 일관성을 위해 모든 컬럼명을 소문자로 변환
    df.columns = [col.lower() for col in df.columns]
    
    # 사용 가능한 모든 컬럼 목록 가져오기
    all_columns = df.columns.tolist()
    
    print("--- 사용 가능한 전체 컬럼 목록 ---")
    # 보기 좋게 5개씩 출력
    for i in range(0, len(all_columns), 5):
        # 각 컬럼 이름을 따옴표로 감싸서 명확하게 표시
        formatted_cols = [f"'{col}'" for col in all_columns[i:i+5]]
        print(", ".join(formatted_cols))
        
    print(f"\n총 {len(all_columns)}개의 컬럼을 찾았습니다.")

if __name__ == "__main__":
    check_all_columns()