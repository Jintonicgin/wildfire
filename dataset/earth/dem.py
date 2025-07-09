import ee
import pandas as pd

# 1. Earth Engine 초기화 (처음 한 번만 인증 필요)
ee.Initialize(project='wildfire-464907')

# 2. CSV 파일 경로 및 로드
CSV_PATH = './location_detail/gangwon_fire_data.csv'
df = pd.read_csv(CSV_PATH)

# 3. USGS SRTM DEM 이미지 로드
dem_dataset = ee.Image('USGS/SRTMGL1_003')

# 4. DEM 값을 저장할 리스트 준비
dem_values = []

# 5. 좌표별로 DEM 값 추출
for idx, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    if pd.isna(lat) or pd.isna(lon):
        dem_values.append(None)
        continue

    point = ee.Geometry.Point(lon, lat)  # GEE 좌표계는 (lon, lat)
    try:
        elevation = dem_dataset.sample(point, scale=30).first().get('elevation').getInfo()
        dem_values.append(elevation)
        print(f"Index {idx}: DEM = {elevation} m")
    except Exception as e:
        print(f"Index {idx}: DEM 조회 실패 - {e}")
        dem_values.append(None)

# 6. 결과를 DataFrame에 추가
df['elevation'] = dem_values

# 7. 일부 결과 출력
print(df[['latitude', 'longitude', 'elevation']].head(20))

# 8. 필요 시 CSV 저장
df.to_csv('gangwon_fire_data_with_dem.csv', index=False, encoding='utf-8-sig')
print("✅ DEM 추가된 CSV 파일 저장 완료: gangwon_fire_data_with_dem.csv")