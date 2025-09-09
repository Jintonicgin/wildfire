import pandas as pd
import numpy as np
import datetime
import requests
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from fwi_calc import fwi_calc

# NASA POWER 파라미터와 사용할 짧은 이름을 매핑
PARAM_MAP = {
    "T2M": "T2M", "RH2M": "RH2M", "WS10M": "WS10M", "WD10M": "WD10M",
    "PRECTOTCORR": "PREC", "PS": "PS", "ALLSKY_SFC_SW_DWN": "SOLAR",
    "WS2M": "WS2M", "WD2M": "WD2M"
}

def _fetch_nasa_hourly_data_for_day(lat, lng, yyyymmdd, max_retry=5):
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": ",".join(PARAM_MAP.keys()), "community": "RE",
        "longitude": lng, "latitude": lat, "start": yyyymmdd, "end": yyyymmdd, "format": "JSON"
    }
    
    for attempt in range(max_retry):
        try:
            # 긴 대기 시간과 재시도 증가
            wait_time = min(3 ** attempt, 30)  # 지수 백오프 (최대 30초)
            if attempt > 0:
                time.sleep(wait_time)
            
            res = requests.get(base_url, params=params, timeout=120)  # 타임아웃 2분으로 증가
            
            if res.status_code == 429:  # Too Many Requests
                print(f"⚠️  Rate limit reached, waiting {wait_time * 2}s...")
                time.sleep(wait_time * 2)
                continue
                
            res.raise_for_status()
            data = res.json().get("properties", {}).get("parameter", {})
            
            if data:  # 데이터가 있으면 성공
                return data
            else:
                print(f"⚠️  Empty data for {yyyymmdd}, retrying...")
                continue
                
        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout for {yyyymmdd} at ({lat}, {lng}), attempt {attempt + 1}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error for {yyyymmdd}: {e}, attempt {attempt + 1}")
            continue
    
    # 모든 재시도 실패 시
    return None

def process_single_row(row_tuple):
    index, row = row_tuple
    lat = row.get('start_latitude')
    lon = row.get('start_longitude')
    target_date = row.get('start_datetime_combined')

    if pd.isna(lat) or pd.isna(lon) or pd.isna(target_date):
        print(f"❌ Row {index}: 필수 데이터 누락 - lat: {lat}, lon: {lon}, date: {target_date}")
        return index, None

    if index % 50 == 0:  # 50개마다 로그 출력
        print(f"📡 Row {index}: 데이터 수집 시작 - ({lat:.4f}, {lon:.4f}) at {target_date}")

    # --- 데이터 수집 및 가공 (순차 처리로 변경) ---
    all_hourly_data = []
    for i in range(8):
        current_date = target_date.date() - datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        try:
            daily_data = _fetch_nasa_hourly_data_for_day(lat, lon, date_str)
            all_hourly_data.append(daily_data)
            if index % 50 == 0:  # 로그 출력 빈도 감소
                if daily_data:
                    print(f"  ✅ {date_str}: {len(daily_data)} 파라미터 수집")
                else:
                    print(f"  ❌ {date_str}: 데이터 수집 실패")
        except Exception as e:
            if index % 50 == 0:  # 로그 출력 빈도 감소
                print(f"  ❌ {date_str} 오류: {e}")
            all_hourly_data.append(None)

    weather_params_short = list(PARAM_MAP.values())
    full_timeseries = {long_name: {} for long_name in PARAM_MAP.keys()}
    for daily_data in all_hourly_data:
        if daily_data: [full_timeseries[param].update(daily_data.get(param, {})) for param in PARAM_MAP.keys()]

    end_time = target_date.replace(minute=0, second=0, microsecond=0)
    start_time = end_time - datetime.timedelta(hours=168)
    hourly_index = pd.to_datetime(pd.date_range(start=start_time, end=end_time, freq='h'))

    df_weather = pd.DataFrame(index=hourly_index)
    for long_name, short_name in PARAM_MAP.items():
        s = pd.Series(full_timeseries[long_name])
        if not s.empty:
            s.index = pd.to_datetime(s.index, format='%Y%m%d%H')
            df_weather[short_name] = s

    df_weather.interpolate(method='time', inplace=True); df_weather.bfill(inplace=True); df_weather.ffill(inplace=True)
    
    # --- 피처 생성 ---
    generated_features = {}
    # 3시간 간격 시점 피처
    for hour_offset in range(0, 169, 3):
        row_data = df_weather.loc[end_time - datetime.timedelta(hours=hour_offset)]
        for param in weather_params_short:
            generated_features[f"{param.lower()}_{hour_offset}h_past"] = row_data.get(param)

    # 7일 & 24시간 통계 피처
    for col in df_weather.columns:
        if df_weather[col].notna().any():
            for stat in ['max', 'min', 'mean', 'std']:
                generated_features[f'{col.lower()}_{stat}_past'] = getattr(df_weather[col], stat)()
    df_weather_24h = df_weather.tail(24)
    for col in df_weather_24h.columns:
        if df_weather_24h[col].notna().any():
            for stat in ['max', 'min', 'mean', 'std']:
                generated_features[f'{col.lower()}_{stat}_24h_past'] = getattr(df_weather_24h[col], stat)()

    # 계절 피처
    month = target_date.month
    generated_features['is_spring'] = 1 if month in [3, 4, 5] else 0
    generated_features['is_summer'] = 1 if month in [6, 7, 8] else 0
    generated_features['is_autumn'] = 1 if month in [9, 10, 11] else 0
    generated_features['is_winter'] = 1 if month in [12, 1, 2] else 0
    
    # --- FWI (Forest Fire Weather Index) 계산 추가 (누락 데이터 처리) ---
    try:
        # 과거 데이터에서 평균값 구하기 (FFMC, DMC, DC 초기값 등) - None 값 처리
        t_0h = generated_features.get('t2m_0h_past') 
        rh_0h = generated_features.get('rh2m_0h_past')
        ws_0h = generated_features.get('ws10m_0h_past')
        prec_0h = generated_features.get('prec_0h_past')
        
        # None 값이나 누락된 데이터에 대한 기본값 설정
        if t_0h is None or not np.isfinite(t_0h):
            t_0h = 15.0  # 기본 온도
        if rh_0h is None or not np.isfinite(rh_0h):
            rh_0h = 50.0  # 기본 습도
        if ws_0h is None or not np.isfinite(ws_0h):
            ws_0h = 3.0   # 기본 풍속
        if prec_0h is None or not np.isfinite(prec_0h):
            prec_0h = 0.0  # 기본 강수량
        
        # 기존 CSV에 있는 데이터들 활용 
        consecutive_dry_days = row.get('consecutive_dry_days_start', 0)
        total_precip_30d = row.get('total_precip_30d_start', 0)
        
        # FWI 계산 (현재 시점)
        fwi_0h = fwi_calc(t_0h, rh_0h, ws_0h, prec_0h, month,
                         consecutive_dry_days=consecutive_dry_days,
                         total_precip_30d=total_precip_30d)
        
        for key, value in fwi_0h.items():
            generated_features[f'{key.lower()}_0h'] = value
            
        # 12시간 평균 FWI 계산 (None 값 처리)
        fwi_values = []
        for h in [0, 3, 6, 9, 12]:
            if f't2m_{h}h_past' in generated_features:
                # None 값 처리
                temp_val = generated_features.get(f't2m_{h}h_past')
                rh_val = generated_features.get(f'rh2m_{h}h_past')
                ws_val = generated_features.get(f'ws10m_{h}h_past')
                prec_val = generated_features.get(f'prec_{h}h_past')
                
                # 기본값으로 대체
                temp_val = temp_val if temp_val is not None and np.isfinite(temp_val) else t_0h
                rh_val = rh_val if rh_val is not None and np.isfinite(rh_val) else rh_0h
                ws_val = ws_val if ws_val is not None and np.isfinite(ws_val) else ws_0h
                prec_val = prec_val if prec_val is not None and np.isfinite(prec_val) else prec_0h
                
                temp_fwi = fwi_calc(
                    temp_val, rh_val, ws_val, prec_val,
                    month, consecutive_dry_days=consecutive_dry_days,
                    total_precip_30d=total_precip_30d
                )
                fwi_values.append(temp_fwi)
                
        if fwi_values:
            for key in ['ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']:
                values = [fwi[key.upper()] for fwi in fwi_values if fwi[key.upper()] != -999]
                if values:
                    generated_features[f'{key}_mean_0_12h'] = np.mean(values)
                else:
                    generated_features[f'{key}_mean_0_12h'] = -999
        
        # 기상 변동성 및 극값 피처 (None 값 처리)
        if 'ws10m_0h_past' in generated_features:
            wind_12h = []
            temp_12h = []
            rh_12h = []
            wd_12h = []
            
            for h in range(0, 13, 3):
                # None 값을 기본값으로 대체
                wind_val = generated_features.get(f'ws10m_{h}h_past')
                temp_val = generated_features.get(f't2m_{h}h_past') 
                rh_val = generated_features.get(f'rh2m_{h}h_past')
                wd_val = generated_features.get(f'wd10m_{h}h_past')
                
                wind_12h.append(wind_val if wind_val is not None and np.isfinite(wind_val) else 0)
                temp_12h.append(temp_val if temp_val is not None and np.isfinite(temp_val) else 0)
                rh_12h.append(rh_val if rh_val is not None and np.isfinite(rh_val) else 0)
                wd_12h.append(wd_val if wd_val is not None and np.isfinite(wd_val) else 0)
            
            generated_features['ws10m_std_0_12h'] = np.std(wind_12h)
            generated_features['t2m_std_0_12h'] = np.std(temp_12h)
            generated_features['rh2m_std_0_12h'] = np.std(rh_12h)
            generated_features['wd10m_std_0_12h'] = np.std(wd_12h)
            generated_features['wd10m_var_0_12h'] = np.var(wd_12h)
            
            generated_features['max_wind_0_12h'] = max(wind_12h)
            generated_features['min_humidity_0_12h'] = min(rh_12h)
            generated_features['max_temp_0_12h'] = max(temp_12h)
            
        # 위험 플래그 피처
        generated_features['high_wind_flag'] = 1 if generated_features.get('ws10m_0h_past', 0) > 15 else 0
        generated_features['low_humidity_flag'] = 1 if generated_features.get('rh2m_0h_past', 100) < 30 else 0
        generated_features['extreme_hot_flag'] = 1 if generated_features.get('t2m_0h_past', 0) > 35 else 0
        generated_features['wind_steady_flag'] = 1 if generated_features.get('wd10m_std_0_12h', 180) < 30 else 0
        
        # 복합 위험 지수
        ws_val = generated_features.get('ws10m_0h_past', 0)
        t2m_val = generated_features.get('t2m_0h_past', 0)
        rh_val = generated_features.get('rh2m_0h_past', 100)
        
        generated_features['dry_windy_combo'] = ws_val * max(0, 100 - rh_val) / 100
        generated_features['hot_dry_combo'] = t2m_val * max(0, 100 - rh_val) / 100
        
    except Exception as e:
        print(f"FWI 계산 오류 for row {index}: {e}")
        # FWI 기본값
        for key in ['ffmc_0h', 'dmc_0h', 'dc_0h', 'isi_0h', 'bui_0h', 'fwi_0h',
                   'ffmc_mean_0_12h', 'dmc_mean_0_12h', 'dc_mean_0_12h', 
                   'isi_mean_0_12h', 'bui_mean_0_12h', 'fwi_mean_0_12h']:
            generated_features[key] = -999

    return index, generated_features

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    io_filename = "final_merged_feature_engineered.csv"
    io_path = os.path.join(script_dir, io_filename)
    
    # 백업 파일 생성
    backup_path = io_path.replace('.csv', '_backup.csv')
    
    try:
        base_df = pd.read_csv(io_path, low_memory=False)
        print(f"✅ 원본 파일 로드 완료: {len(base_df)} 행")
        
        # 백업 생성 (기존 백업이 없는 경우만)
        if not os.path.exists(backup_path):
            base_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            print(f"🛡️  백업 파일 생성: {backup_path}")
        else:
            print(f"🛡️  기존 백업 파일 사용: {backup_path}")
            
    except FileNotFoundError:
        print(f"❌ Error: 입력 파일을 찾을 수 없습니다: {io_path}")
        return

    base_df.columns = [col.lower() for col in base_df.columns]

    lat_col, lon_col = 'start_latitude', 'start_longitude'
    year_col, month_col, day_col, time_col = 'startyear', 'startmonth', 'startday', 'starttime'

    required_cols = [lat_col, lon_col, year_col, month_col, day_col, time_col]
    if not all(col in base_df.columns for col in required_cols):
        print(f"❌ Error: 필요한 컬럼({required_cols})이 파일에 없습니다.")
        return

    # --- 날짜 및 시간 조합 로직 (이미 형식화된 시간 처리) ---
    datetime_str = base_df[year_col].astype(str) + '-' + \
                   base_df[month_col].astype(str) + '-' + \
                   base_df[day_col].astype(str) + ' ' + \
                   base_df[time_col].astype(str)
    base_df['start_datetime_combined'] = pd.to_datetime(datetime_str, errors='coerce')
    
    print(f"✅ 날짜 변환 완료. 예시: {base_df['start_datetime_combined'].iloc[0]}")
    failed_dates = base_df['start_datetime_combined'].isna().sum()
    if failed_dates > 0:
        print(f"⚠️  {failed_dates}개 행의 날짜 변환에 실패했습니다.")

    # 전체 데이터 처리
    tasks = [row for row in base_df.iterrows()]
    print(f"🚀 전체 데이터 처리 모드: {len(tasks)}개 행을 처리합니다.")
    
    # --- 병렬 처리 with 데이터 안전성 보장 ---
    print(f"병렬 처리로 {len(tasks)}개 행의 피처 생성을 시작합니다...")
    
    # 결과 저장용 딕셔너리 (스레드 안전)
    results_dict = {}
    
    with ProcessPoolExecutor(max_workers=min(5, os.cpu_count())) as executor:  # API 부하 감소를 위해 워커 수 더 감소
        # 모든 작업을 한 번에 제출
        future_to_index = {executor.submit(process_single_row, task): task[0] for task in tasks}
        
        # 완료된 작업들을 처리
        for future in tqdm(as_completed(future_to_index), total=len(tasks), desc="피처 생성 진행률"):
            idx = future_to_index[future]
            try:
                result_idx, new_features = future.result()
                if new_features:
                    results_dict[result_idx] = new_features
            except Exception as e:
                print(f"❌ Row {idx} 처리 오류: {e}")
                continue
    
    # 한 번에 모든 결과를 DataFrame에 적용 (메모리 효율적)
    print("결과를 DataFrame에 적용중...")
    all_new_columns = set()
    for features in results_dict.values():
        all_new_columns.update(features.keys())
    
    # 새 컬럼들을 효율적으로 생성 (pd.concat 사용)
    new_columns_df = pd.DataFrame(index=base_df.index)
    for col in all_new_columns:
        safe_col_name = col if col.endswith('_past') or col in ['is_spring', 'is_summer', 'is_autumn', 'is_winter'] \
                       or any(keyword in col for keyword in ['_0h', '_mean_0_12h', '_std_0_12h', '_flag', '_combo']) \
                       else f"{col}_new"
        if safe_col_name not in base_df.columns:
            new_columns_df[safe_col_name] = np.nan
    
    # 한 번에 새 컬럼들을 추가
    base_df = pd.concat([base_df, new_columns_df], axis=1)
    
    # 배치로 결과 적용
    for idx, new_features in results_dict.items():
        for col, value in new_features.items():
            safe_col_name = col if col.endswith('_past') or col in ['is_spring', 'is_summer', 'is_autumn', 'is_winter'] \
                           or any(keyword in col for keyword in ['_0h', '_mean_0_12h', '_std_0_12h', '_flag', '_combo']) \
                           else f"{col}_new"
            base_df.loc[idx, safe_col_name] = value
    
    # 임시로 사용한 날짜 컬럼 삭제
    if 'start_datetime_combined' in base_df.columns:
        base_df.drop(columns=['start_datetime_combined'], inplace=True)

    # 최종 저장 전 검증
    print(f"최종 DataFrame 크기: {base_df.shape}")
    print(f"추가된 새 컬럼 수: {len(all_new_columns)}")
    
    base_df.to_csv(io_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 모든 작업 완료! 신규 피처가 추가/업데이트되어 {io_filename} 파일에 저장되었습니다.")
    print(f"📊 최종 결과: {len(base_df)} 행, {len(base_df.columns)} 컬럼")
    print(f"🛡️  문제 발생 시 백업 파일을 사용하세요: {backup_path}")

if __name__ == "__main__":
    main()