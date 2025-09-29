import datetime
import numpy as np
import pandas as pd
import requests
from wildfire.ML.fwi_calc import fwi_calc

# NASA POWER 파라미터와 사용할 짧은 이름을 매핑
PARAM_MAP = {
    "T2M": "T2M", "RH2M": "RH2M", "WS10M": "WS10M", "WD10M": "WD10M",
    "PRECTOTCORR": "PREC", "PS": "PS", "ALLSKY_SFC_SW_DWN": "SOLAR",
    "WS2M": "WS2M", "WD2M": "WD2M"
}


def _fetch_nasa_raw_hourly_data_for_day(lat, lng, yyyymmdd, max_retry=3):
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters={','.join(PARAM_MAP.keys())}&community=RE&longitude={lng}&latitude={lat}&start={yyyymmdd}&end={yyyymmdd}&format=JSON"
    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            return res.json().get("properties", {}).get("parameter", {})
        except requests.exceptions.RequestException as e:
            if attempt == max_retry - 1:
                print(f"❌ NASA API 호출 최종 실패: {e}")
                return None


def fetch_nasa_daily_precip(lat, lng, start_date, end_date, max_retry=3):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=RE&longitude={lng}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            return res.json().get("properties", {}).get("parameter", {})
        except Exception:
            if attempt == max_retry - 1:
                return None


def make_precip_features(lat, lng, end_dt, periods=[7, 14, 30, 60, 90]):
    res = {}
    max_ndays = max(periods)
    start_date_str = (end_dt - datetime.timedelta(days=max_ndays - 1)).strftime("%Y%m%d")
    end_date_str = end_dt.strftime("%Y%m%d")
    full_precip = fetch_nasa_daily_precip(lat, lng, start_date_str, end_date_str) or {}
    precip_data = full_precip.get("PRECTOTCORR", {})

    if not precip_data:
        for p in periods:
            res[f"total_prec_{p}d_start"] = 0.0
            res[f"dry_days_{p}d_start"] = p
            res[f"total_prec_{p}d_start_past"] = 0.0
            res[f"dry_days_{p}d_start_past"] = p
        res["consecutive_dry_days_start"] = max_ndays
        res["consecutive_dry_days_start_past"] = max_ndays
        return res

    sorted_precip = sorted(precip_data.items(), key=lambda item: datetime.datetime.strptime(item[0], '%Y%m%d'),
                           reverse=True)

    for ndays in periods:
        period_data = [float(v) for k, v in sorted_precip[:ndays] if v != -999]
        arr = np.array(period_data)
        total_prec = float(np.sum(arr)) if arr.size > 0 else 0.0
        dry_days = int(np.sum(arr < 1)) if arr.size > 0 else ndays
        res[f"total_prec_{ndays}d_start"] = total_prec
        res[f"dry_days_{ndays}d_start"] = dry_days
        res[f"total_prec_{ndays}d_start_past"] = total_prec
        res[f"dry_days_{ndays}d_start_past"] = dry_days

    cons = 0
    all_days_data = [float(v) for k, v in sorted_precip if v != -999]
    for v in all_days_data:
        if v < 1:
            cons += 1
        else:
            break
    res["consecutive_dry_days_start"] = int(cons)
    res["consecutive_dry_days_start_past"] = int(cons)
    return res


def generate_weather_statistics_features(df_weather, end_dt):
    """기상 데이터로부터 통계 피처들을 생성합니다."""
    stats_features = {}

    # 파라미터별로 통계 계산
    weather_params = ['T2M', 'RH2M', 'WS10M', 'WD10M', 'PREC', 'PS', 'SOLAR']

    for param in weather_params:
        if param in df_weather.columns:
            values = df_weather[param].dropna()
            if len(values) > 0:
                # 전체 기간 통계 (168시간)
                stats_features[f"{param.lower()}_max_past"] = float(values.max())
                stats_features[f"{param.lower()}_min_past"] = float(values.min())
                stats_features[f"{param.lower()}_mean_past"] = float(values.mean())
                stats_features[f"{param.lower()}_std_past"] = float(values.std())

                # 24시간 통계
                day_values = values[-24:] if len(values) >= 24 else values
                stats_features[f"{param.lower()}_max_24h_past"] = float(day_values.max())
                stats_features[f"{param.lower()}_min_24h_past"] = float(day_values.min())
                stats_features[f"{param.lower()}_mean_24h_past"] = float(day_values.mean())
                stats_features[f"{param.lower()}_std_24h_past"] = float(day_values.std())

                # 12시간 통계 (추가)
                half_day_values = values[-12:] if len(values) >= 12 else values
                stats_features[f"max_temp_0_12h"] = float(values[-12:].max()) if param == 'T2M' else None
                stats_features[f"min_humidity_0_12h"] = float(values[-12:].min()) if param == 'RH2M' else None
                stats_features[f"max_wind_0_12h"] = float(values[-12:].max()) if param == 'WS10M' else None
            else:
                # 데이터가 없는 경우 기본값
                for suffix in ['_max_past', '_min_past', '_mean_past', '_std_past',
                               '_max_24h_past', '_min_24h_past', '_mean_24h_past', '_std_24h_past']:
                    stats_features[f"{param.lower()}{suffix}"] = -999

    # 특별 통계 피처들
    if 'T2M' in df_weather.columns and 'RH2M' in df_weather.columns:
        # 체감온도 계산 (간단한 버전)
        t_vals = df_weather['T2M'].dropna()
        rh_vals = df_weather['RH2M'].dropna()
        if len(t_vals) > 0 and len(rh_vals) > 0:
            # 온도와 습도의 상관관계
            stats_features['temp_humidity_correlation'] = float(np.corrcoef(t_vals[-min(len(t_vals), len(rh_vals)):],
                                                                            rh_vals[-min(len(t_vals), len(rh_vals)):])[
                                                                    0, 1])

    # None 값들을 적절한 기본값으로 대체
    for key, value in stats_features.items():
        if value is None or pd.isna(value):
            stats_features[key] = -999

    return stats_features


def generate_weather_trend_features(df_weather, end_dt):
    """기상 변화 트렌드 피처들을 생성합니다."""
    trend_features = {}

    weather_params = ['T2M', 'RH2M', 'WS10M', 'PREC']

    for param in weather_params:
        if param in df_weather.columns:
            values = df_weather[param].dropna()
            if len(values) >= 6:
                # 6시간 변화율
                trend_features[f"{param.lower()}_trend_6h"] = float(values.iloc[-1] - values.iloc[-6])
                # 12시간 변화율
                if len(values) >= 12:
                    trend_features[f"{param.lower()}_trend_12h"] = float(values.iloc[-1] - values.iloc[-12])
                # 24시간 변화율
                if len(values) >= 24:
                    trend_features[f"{param.lower()}_trend_24h"] = float(values.iloc[-1] - values.iloc[-24])

                # 변동성 지수 (최근 24시간 표준편차)
                recent_24h = values[-24:] if len(values) >= 24 else values
                trend_features[f"{param.lower()}_volatility_24h"] = float(recent_24h.std())
            else:
                # 데이터가 부족한 경우
                for suffix in ['_trend_6h', '_trend_12h', '_trend_24h', '_volatility_24h']:
                    trend_features[f"{param.lower()}{suffix}"] = 0

    return trend_features


def generate_fire_risk_composite_features(current_weather, precip_feats):
    """복합 화재 위험도 피처들을 생성합니다."""
    composite_features = {}

    # 기본 값들
    temp = current_weather.get('T2M', 20)
    humidity = current_weather.get('RH2M', 50)
    wind = current_weather.get('WS10M', 5)
    precip = current_weather.get('PREC', 0)

    dry_days_7d = precip_feats.get('dry_days_7d_start', 7)
    dry_days_30d = precip_feats.get('dry_days_30d_start', 30)
    total_precip_30d = precip_feats.get('total_prec_30d_start', 0)

    # 1. 건조-바람 복합 지수
    dry_factor = max(0, (100 - humidity) / 100)
    wind_factor = min(wind / 20, 1.0)
    composite_features['dry_windy_combo'] = float(dry_factor * wind_factor)

    # 2. 연료 건조도 지수
    temp_factor = max(0, (temp - 15) / 25) if temp > 15 else 0
    drought_factor = min(dry_days_30d / 30, 1.0)
    composite_features['fuel_dryness_index'] = float((temp_factor + drought_factor + dry_factor) / 3)

    # 3. 확산 잠재력 지수
    spread_index = (wind_factor * 0.4 + dry_factor * 0.3 + temp_factor * 0.3)
    composite_features['potential_spread_index'] = float(spread_index)

    # 4. 30일 건조/강수 비율
    if total_precip_30d > 0:
        composite_features['dry_to_rain_ratio_30d'] = float(dry_days_30d / max(total_precip_30d, 0.1))
    else:
        composite_features['dry_to_rain_ratio_30d'] = 30.0

    # 5. 극한 기상 플래그들
    composite_features['extreme_temp_flag'] = 1 if temp > 35 or temp < -10 else 0
    composite_features['extreme_wind_flag'] = 1 if wind > 15 else 0
    composite_features['extreme_dry_flag'] = 1 if humidity < 20 else 0
    composite_features['no_rain_week_flag'] = 1 if dry_days_7d >= 7 else 0

    return composite_features


def fetch_all_weather_features(lat, lon, timestamp, offset_days=0):
    target_dt = timestamp - datetime.timedelta(days=offset_days)
    end_dt = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"🌤️  기상 데이터 수집 중... ({lat:.4f}, {lon:.4f})")

    # 1. 강수량 피처 생성
    precip_feats = make_precip_features(lat, lon, end_dt)
    print(f"✅ 강수량 피처 {len(precip_feats)}개 생성")

    # 2. 시간별 기상 데이터 수집
    all_hourly_data = []
    for i in range(8):
        daily_data = _fetch_nasa_raw_hourly_data_for_day(lat, lon,
                                                         (end_dt - datetime.timedelta(days=i)).strftime("%Y%m%d"))
        all_hourly_data.append(daily_data)

    full_timeseries = {param: {} for param in PARAM_MAP.keys()}
    for daily_data in all_hourly_data:
        if daily_data: [full_timeseries[param].update(daily_data.get(param, {})) for param in PARAM_MAP.keys()]

    start_time = end_dt - datetime.timedelta(hours=168)
    hourly_index = pd.to_datetime(pd.date_range(start=start_time, end=end_dt, freq='h'))
    df_weather = pd.DataFrame(index=hourly_index)

    for long_name, short_name in PARAM_MAP.items():
        s = pd.Series(full_timeseries[long_name])
        if not s.empty:
            s.index = pd.to_datetime(s.index, format='%Y%m%d%H')
            s = pd.to_numeric(s, errors='coerce').replace(-999, np.nan)
            df_weather[short_name] = s
        else:
            df_weather[short_name] = np.nan

    df_weather.interpolate(method='time', inplace=True)
    df_weather.bfill(inplace=True)
    df_weather.ffill(inplace=True)
    df_weather.fillna(0, inplace=True)

    # 3. 기본 시간별 피처 생성
    generated_features = {}
    weather_params_short = list(PARAM_MAP.values())

    for hour_offset in range(0, 169, 3):
        row_data = df_weather.loc[end_dt - datetime.timedelta(hours=hour_offset)]
        for param in weather_params_short:
            val = row_data.get(param)
            feature_name = f"{param.lower()}_{hour_offset}h"
            generated_features[feature_name] = val
            generated_features[f"{feature_name}_past"] = val

    print(f"✅ 시간별 기상 피처 {len([k for k in generated_features.keys() if 'h' in k])}개 생성")

    # 4. FWI 계산
    current_weather = df_weather.iloc[-1].to_dict()
    fwi_inputs = {key: current_weather.get(key) for key in ["T2M", "RH2M", "WS10M", "PREC"]}
    if any(pd.isna(v) for v in fwi_inputs.values()):
        fwi = {k: -999 for k in ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]}
        print("⚠️  FWI 계산 실패 - 입력값 부족")
    else:
        fwi = fwi_calc(T=fwi_inputs["T2M"], RH=fwi_inputs["RH2M"], W=fwi_inputs["WS10M"], P=fwi_inputs["PREC"],
                       month=end_dt.month)
        print("✅ FWI 지수 계산 완료")

    for k, v in fwi.items():
        generated_features[f"{k.lower()}_0h"] = v
        generated_features[f"{k.lower()}_0h_past"] = v

    # 5. 통계 피처 생성
    print("📊 통계 피처 생성 중...")
    stats_features = generate_weather_statistics_features(df_weather, end_dt)
    generated_features.update(stats_features)
    print(f"✅ 통계 피처 {len(stats_features)}개 생성")

    # 6. 트렌드 피처 생성
    print("📈 트렌드 피처 생성 중...")
    trend_features = generate_weather_trend_features(df_weather, end_dt)
    generated_features.update(trend_features)
    print(f"✅ 트렌드 피처 {len(trend_features)}개 생성")

    # 7. 복합 화재 위험도 피처 생성
    print("🔥 복합 위험도 피처 생성 중...")
    composite_features = generate_fire_risk_composite_features(current_weather, precip_feats)
    generated_features.update(composite_features)
    print(f"✅ 복합 위험도 피처 {len(composite_features)}개 생성")

    # 8. 계절 피처 생성
    month = end_dt.month
    generated_features['is_spring'] = 1 if month in [3, 4, 5] else 0
    generated_features['is_summer'] = 1 if month in [6, 7, 8] else 0
    generated_features['is_autumn'] = 1 if month in [9, 10, 11] else 0
    generated_features['is_winter'] = 1 if month in [12, 1, 2] else 0

    # 날짜 피처 추가
    generated_features['startday'] = end_dt.day
    generated_features['startmonth'] = end_dt.month
    generated_features['startyear'] = end_dt.year

    generated_features.update(precip_feats)
    generated_features["success"] = True

    print(f"🎉 총 기상 피처 생성 완료: {len(generated_features)}개")
    return generated_features
