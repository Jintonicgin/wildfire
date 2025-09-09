import ee
import pandas as pd
from datetime import datetime

ee.Initialize(project='wildfire-464907')  

modis_lc_ic = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')

def get_forest_cover_ratio(image, geometry, scale=500):
    freq_hist = image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9
    ).get('LC_Type1')

    freq_hist_dict = freq_hist.getInfo() if freq_hist else None
    if not freq_hist_dict:
        return None

    total_pixels = sum(freq_hist_dict.values())
    forest_classes = [1, 2, 3, 4, 5]  
    forest_pixels = sum(freq_hist_dict.get(str(c), 0) for c in forest_classes)
    return (forest_pixels / total_pixels) * 100 if total_pixels > 0 else None

def get_forest_type_and_cover(lon, lat, year):
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(5000)

    ic_filtered = modis_lc_ic.filterDate(f'{year}-01-01', f'{year}-12-31').filterBounds(region)
    if ic_filtered.size().getInfo() == 0:
        return None, None

    image = ic_filtered.first()
    if image is None:
        return None, None

    mode_img = image.reduceNeighborhood(
        reducer=ee.Reducer.mode(),
        kernel=ee.Kernel.square(2),
        skipMasked=True
    )
    try:
        forest_type = mode_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=500,
            maxPixels=1e9
        ).get('LC_Type1_mode').getInfo()
    except:
        forest_type = None

    forest_cover = get_forest_cover_ratio(image, region)
    return forest_type, forest_cover

df = pd.read_csv('gangwon_fire_with_forest_cover.csv')

for idx, row in df.iterrows():
    fire_date = pd.to_datetime(row['fire_start_date'])
    year = fire_date.year

    if year == 2024:
        if pd.isna(row.get('forest_type_final')) or pd.isna(row.get('forest_cover_final_percent')):
            lon = row['longitude']
            lat = row['latitude']

            ft, fc = get_forest_type_and_cover(lon, lat, 2024)

            if ft is None or fc is None:
                print(f"[{idx}] 24년 데이터 없음, 23년 데이터로 대체 시도")
                ft, fc = get_forest_type_and_cover(lon, lat, 2023)

            df.at[idx, 'forest_type_final'] = ft
            df.at[idx, 'forest_cover_final_percent'] = fc

            print(f"[{idx}] 24년 데이터 업데이트 - 산림유형: {ft}, 산림율: {fc}")
        else:
            print(f"[{idx}] 24년 데이터 이미 존재, 스킵")

df.to_csv('gangwon_fire_data_updated.csv', index=False)
print("✅ 24년 데이터만 빈 부분 채워서 저장 완료")