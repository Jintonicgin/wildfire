import ee
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd

# GEE 인증 및 초기화
try:
    ee.Initialize(project='wildfire-464907')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='wildfire-464907')

# GeoJSON 로딩
geojson_path = "/Users/mmymacymac/Developer/Projects/WildFire_projects/wildfire/wildfire/static/data/gangwondo_ssg.geojson"
gdf = gpd.read_file(geojson_path)
print("📌 GeoJSON 컬럼 목록:", gdf.columns)

# GEE 이미지 정의
elevation = ee.Image("USGS/SRTMGL1_003")
slope = ee.Terrain.slope(elevation)
treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10").select("treecover2000")

# 산지 조건 마스크 (고도, 경사, 수목 조건)
mountain_mask = elevation.gt(100).And(slope.gt(3)).And(treecover.gt(20))

features = []

for _, row in gdf.iterrows():
    region_name = row['title']
    region_geom = ee.Geometry(mapping(row['geometry']))

    masked = mountain_mask.clip(region_geom).selfMask()

    # 조건을 만족하는 픽셀의 중심점 벡터화
    vectors = masked.reduceToVectors(
        geometry=region_geom,
        geometryType='centroid',
        scale=90,
        maxPixels=1e8
    )

    sample = vectors.randomColumn().sort('random').first()

    try:
        coords = ee.Feature(sample).geometry().coordinates().getInfo()
        features.append({
            "region": region_name,
            "lat": coords[1],
            "lng": coords[0],
            "source": "mountain"
        })
        print(f"✅ {region_name} → 산지 좌표 추출 완료")
    except Exception:
        centroid = row['geometry'].centroid
        features.append({
            "region": region_name,
            "lat": centroid.y,
            "lng": centroid.x,
            "source": "centroid"
        })
        print(f"❗️ {region_name} → 산지 없음, 중심점 사용")

# 결과 저장
df = pd.DataFrame(features)
df.to_csv("gangwon_mountain_points.csv", index=False)
print("📁 CSV 저장 완료: gangwon_mountain_points.csv")