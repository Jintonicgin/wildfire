import ee
import time

# 인증 및 초기화
ee.Authenticate()
ee.Initialize(project='wildfire-464907')

print("✅ 시작: NDVI + 기후변수 수집")

# 강원도 geometry
region = ee.FeatureCollection("projects/wildfire-464907/assets/gangwon-do")

# 연도 범위
start_year = 2001
end_year = 2023

# NDVI & 기후 컬렉션
ndvi_coll = ee.ImageCollection("MODIS/006/MOD13Q1")
climate_coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

# 연도별 처리 함수
def process_year(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)

    # NDVI
    ndvi = ndvi_coll.filterDate(start, end).select("NDVI").mean().multiply(0.0001)

    # 기후 변수들 평균 or 합
    climate = climate_coll.filterDate(start, end)
    temp = climate.select("temperature_2m").mean()
    precip = climate.select("total_precipitation_sum").sum()
    pressure = climate.select("surface_pressure").mean()
    wind_u = climate.select("u_component_of_wind_10m").mean()
    wind_v = climate.select("v_component_of_wind_10m").mean()
    wind_speed = wind_u.hypot(wind_v)

    # 결합
    combined = ndvi.addBands([
        temp.rename("mean_temp"),
        precip.rename("total_precip"),
        pressure.rename("mean_pressure"),
        wind_speed.rename("mean_wind")
    ])

    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region.geometry(),
        scale=500,
        maxPixels=1e13
    )

    return ee.Feature(None, stats).set("year", year)

# 연도 리스트 -> FeatureCollection
years = ee.List.sequence(start_year, end_year)
features = years.map(lambda y: process_year(ee.Number(y).toInt()))
result_fc = ee.FeatureCollection(features)

# Export to Drive
task = ee.batch.Export.table.toDrive(
    collection=result_fc,
    description="Export_NDVI_Climate",
    fileFormat="CSV",
    folder="EarthEngine"
)

task.start()
print("🚀 Export started...")

# 상태 체크
while task.active():
    print("⏳ Export task is running...")
    time.sleep(10)

# 결과
status = task.status()
if status['state'] == 'COMPLETED':
    print("✅ Export completed successfully!")
else:
    print(f"❌ Export failed: {status}")