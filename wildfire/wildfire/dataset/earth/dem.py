import ee
import pandas as pd

ee.Initialize(project='wildfire-464907')

XLSX_PATH = './location_detail/gangwon_fire_data_with_coords.xlsx'
df = pd.read_excel(XLSX_PATH)

dem_dataset = ee.Image('USGS/SRTMGL1_003')

# 경사도, 방위각 계산 (단위: 경사도는 도 단위, 방위각도는 도 단위)
terrain = ee.Terrain.products(dem_dataset)
slope_img = terrain.select('slope')   # 경사도
aspect_img = terrain.select('aspect') # 방위각

dem_values = []
slope_values = []
aspect_values = []

for idx, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    if pd.isna(lat) or pd.isna(lon):
        dem_values.append(None)
        slope_values.append(None)
        aspect_values.append(None)
        continue

    point = ee.Geometry.Point(lon, lat)
    try:
        sample = dem_dataset.sample(point, scale=30).first()
        slope_sample = slope_img.sample(point, scale=30).first()
        aspect_sample = aspect_img.sample(point, scale=30).first()

        if sample and slope_sample and aspect_sample:
            elevation = sample.get('elevation').getInfo()
            slope = slope_sample.get('slope').getInfo()
            aspect = aspect_sample.get('aspect').getInfo()

            dem_values.append(elevation)
            slope_values.append(slope)
            aspect_values.append(aspect)

            print(f"Index {idx}: DEM={elevation}m, Slope={slope}°, Aspect={aspect}°")
        else:
            dem_values.append(None)
            slope_values.append(None)
            aspect_values.append(None)
            print(f"Index {idx}: 데이터 없음")
    except Exception as e:
        dem_values.append(None)
        slope_values.append(None)
        aspect_values.append(None)
        print(f"Index {idx}: 조회 실패 - {e}")

df['elevation'] = dem_values
df['slope'] = slope_values
df['aspect'] = aspect_values

print(df[['latitude', 'longitude', 'elevation', 'slope', 'aspect']].head(20))

df.to_csv('gangwon_fire_data_with_dem_slope_aspect.csv', index=False, encoding='utf-8-sig')
print("✅ DEM, 경사도, 방위각 포함된 CSV 저장 완료")