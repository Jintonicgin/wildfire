import ee
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

# 1. Earth Engine 초기화 (처음 한 번은 ee.Authenticate() 실행)
ee.Initialize(project='wildfire-464907')

# 2. MODIS NDVI 컬렉션 (MOD13Q1)
modis_ndvi_ic = ee.ImageCollection('MODIS/061/MOD13Q1').select('NDVI')

# 3. MODIS Land Cover (MCD12Q1) - IGBP 분류
modis_lc_ic = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')

def get_latest_ndvi_pre_fire(lon, lat, fire_date_str):
    geometry = ee.Geometry.Point(lon, lat)
    fire_date = datetime.strptime(fire_date_str, "%Y-%m-%d")
    start_date = fire_date - timedelta(days=30)
    end_date = fire_date - timedelta(days=1)

    filtered_ic = (modis_ndvi_ic
                   .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                   .filterBounds(geometry)
                   .sort('system:time_start', False))

    size = filtered_ic.size().getInfo()
    if size == 0:
        print(f"[{lon}, {lat}] 기간 {start_date}~{end_date}에 NDVI 이미지 없음")
        return None

    latest_img = filtered_ic.first()
    if latest_img is None:
        print(f"[{lon}, {lat}] 최신 NDVI 이미지 없음")
        return None

    stats = latest_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=250
    ).get('NDVI').getInfo()

    if stats is not None:
        return stats / 10000.0  # NDVI 정규화
    else:
        return None

def get_forest_type_mode_5x5(lon, lat, year):
    try:
        geometry = ee.Geometry.Point(lon, lat).buffer(5000)  # 5km 버퍼

        filtered_ic = ee.ImageCollection("MODIS/061/MCD12Q1") \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .filterBounds(geometry)

        if filtered_ic.size().getInfo() == 0:
            print(f"[{lon}, {lat}] {year}년 산림유형 이미지 없음")
            return None

        image = filtered_ic.first()
        if image is None:
            print(f"[{lon}, {lat}] 산림유형 이미지 없음")
            return None

        mode_img = image.select('LC_Type1').reduceNeighborhood(
            reducer=ee.Reducer.mode(),
            kernel=ee.Kernel.square(2),  # 5x5 window
            inputWeight=None,
            skipMasked=True
        )

        val = mode_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=geometry.centroid(),
            scale=500,
            maxPixels=1e8
        ).get('LC_Type1_mode').getInfo()

        if val is not None:
            return int(val)
        else:
            print(f"[{lon}, {lat}] 산림유형 모드 값 없음")
            return None
    except Exception as e:
        print(f"[{lon}, {lat}] 산림유형 오류: {e}")
        return None
    
# 4. 데이터 불러오기 (경로 조정 필요)
df = pd.read_csv('gangwon_fire_data_with_climate_hourly_segmented.csv')

ndvi_pre_fire_list = []
forest_type_mode_list = []

for idx, row in df.iterrows():
    lon = row['longitude']
    lat = row['latitude']
    fire_date_str = row['fire_start_date']

    try:
        ndvi_pre = get_latest_ndvi_pre_fire(lon, lat, fire_date_str)
    except Exception as e:
        print(f"[{idx}] NDVI 오류: {e}")
        ndvi_pre = None

    try:
        fire_year = int(fire_date_str.split('-')[0])
        forest_type_mode = get_forest_type_mode_5x5(lon, lat, fire_year)
    except Exception as e:
        print(f"[{idx}] 산림유형 오류: {e}")
        forest_type_mode = None

    ndvi_pre_fire_list.append(ndvi_pre)
    forest_type_mode_list.append(forest_type_mode)

    print(f"[{idx}] NDVI: {ndvi_pre}, 산림유형 최빈값: {forest_type_mode}")

df['ndvi_pre_fire_latest'] = ndvi_pre_fire_list
df['forest_type_mode_5km'] = forest_type_mode_list

df.to_csv('gangwon_fire_data_with_ndvi_and_forest_type_mode.csv', index=False, encoding='utf-8-sig')

print("✅ NDVI 및 산림유형 최빈값(5km 버퍼) 추출 완료 및 저장")