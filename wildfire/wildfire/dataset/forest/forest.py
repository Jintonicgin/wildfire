import ee
import pandas as pd

# 1. Earth Engine 초기화 (최초 1회만)
ee.Initialize(project='wildfire-464907')

def calculate_forest_cover(lon, lat, year, buffer_radius=5000):
    region = ee.Geometry.Point(lon, lat).buffer(buffer_radius)

    landcover_ic = ee.ImageCollection('MODIS/061/MCD12Q1') \
                    .filter(ee.Filter.calendarRange(year, year, 'year')) \
                    .filterBounds(region)

    img = landcover_ic.first()
    if img is None:
        print(f"{year}년 데이터 없음 (위도:{lat}, 경도:{lon})")
        return None

    lc_img = img.select('LC_Type1')

    try:
        result = lc_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=500,
            maxPixels=1e9
        ).getInfo()
        print(f"[{year}] reduceRegion 결과 keys: {list(result.keys())}")

        freq_hist = result.get('LC_Type1')
        if freq_hist is None:
            print(f"[{year}] 'LC_Type1' 키 없음 (위도:{lat}, 경도:{lon})")
            return None

        total_pixels = sum(freq_hist.values())
        forest_classes = ['1', '2', '3', '4', '5']  # 산림 클래스 코드(문자열형태)
        forest_pixels = sum(freq_hist.get(k, 0) for k in forest_classes)

    except Exception as e:
        print(f"reduceRegion 호출 오류 (위도:{lat}, 경도:{lon}, 연도:{year}): {e}")
        return None

    if total_pixels == 0:
        print(f"픽셀 수 0 (위도:{lat}, 경도:{lon}, 연도:{year})")
        return None

    forest_ratio = forest_pixels / total_pixels
    forest_percent = forest_ratio * 100
    return forest_percent

# 2. 엑셀 파일 불러오기 (경로는 환경에 맞게 조정)
input_path = './gangwon_fire_data_with_ndvi_and_forest_type_mode.csv'
df = pd.read_csv(input_path)

forest_cover_list = []

for idx, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    # 연도는 발생일(fire_start_date)에서 추출
    fire_date = pd.to_datetime(row['fire_start_date'])
    year = fire_date.year

    try:
        forest_cover = calculate_forest_cover(lon, lat, year)
    except Exception as e:
        print(f"[{idx}] 오류 발생: {e}")
        forest_cover = None
    
    forest_cover_list.append(forest_cover)
    print(f"[{idx}] 위도:{lat}, 경도:{lon}, 연도:{year} 산림율: {forest_cover}")

# 3. 결과 컬럼 추가
df['forest_cover_5km_percent'] = forest_cover_list

# 4. 결과 확인 및 필요시 저장
print(df[['latitude', 'longitude', 'fire_start_date', 'forest_cover_5km_percent']].head())
df.to_csv('gangwon_fire_with_forest_cover.csv', index=False)