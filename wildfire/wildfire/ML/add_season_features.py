import pandas as pd
import os

def add_or_update_season_features():
    """
    final_merged_feature_engineered.csv 파일을 읽어
    계절 피처를 추가하거나 업데이트합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    io_filename = "final_merged_feature_engineered.csv"
    io_path = os.path.join(script_dir, io_filename)

    try:
        df = pd.read_csv(io_path, low_memory=False)
        print(f"✅ '{io_path}' 파일을 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ Error: 입력 파일을 찾을 수 없습니다. 경로를 확인하세요: {io_path}")
        return

    # 대소문자 일관성을 위해 컬럼명을 소문자로 변환
    df.columns = [col.lower() for col in df.columns]

    # 날짜 기준으로 사용할 컬럼 확인
    if 'endmonth' in df.columns:
        date_col = 'endmonth'
    elif 'startyear' in df.columns and 'startmonth' in df.columns and 'startday' in df.columns:
        # endtime이 없으면 start... 컬럼으로 date_combined 생성
        df['date_combined'] = pd.to_datetime(
            df[['startyear', 'startmonth', 'startday']].rename(
                columns={'startyear': 'year', 'startmonth': 'month', 'startday': 'day'}
            ), errors='coerce')
        date_col = 'date_combined'
    else:
        print("❌ Error: 날짜 정보를 담고 있는 컬럼(endtime 또는 startyear/month/day)을 찾을 수 없습니다.")
        return

    # 날짜 컬럼을 datetime 객체로 변환 (오류 발생 시 NaT으로)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # 날짜 정보가 없는 행이 있는지 확인
    if df[date_col].isna().any():
        print(f"⚠️ 경고: 날짜 정보를 변환할 수 없는 행이 {df[date_col].isna().sum()}개 있습니다. 해당 행의 계절 정보는 0으로 처리됩니다.")

    # 계절 피처 생성
    print("계절 피처를 생성 또는 업데이트합니다...")
    df['is_spring'] = df[date_col].dt.month.isin([3, 4, 5]).astype(int)
    df['is_summer'] = df[date_col].dt.month.isin([6, 7, 8]).astype(int)
    df['is_autumn'] = df[date_col].dt.month.isin([9, 10, 11]).astype(int)
    df['is_winter'] = df[date_col].dt.month.isin([12, 1, 2]).astype(int)
    print("✅ 계절 피처 생성 완료: is_spring, is_summer, is_autumn, is_winter")

    # 임시로 사용한 날짜 컬럼 삭제
    if 'date_combined' in df.columns:
        df.drop(columns=['date_combined'], inplace=True)

    # 결과 저장 (기존 파일 덮어쓰기)
    try:
        df.to_csv(io_path, index=False, encoding='utf-8-sig')
        print(f"\n🎉 모든 작업 완료! 계절 피처가 추가되어 {io_filename} 파일에 저장되었습니다.")
    except Exception as e:
        print(f"\n❌ 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    add_or_update_season_features()
