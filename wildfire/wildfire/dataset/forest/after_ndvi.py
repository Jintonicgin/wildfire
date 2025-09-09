import ee
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time

ee.Initialize(project='wildfire-464907')

INPUT_CSV = "./gangwon_fire_data_with_climate_hourly_parallel.csv"
OUTPUT_CSV = "./gangwon_fire_data_with_ndvi.csv"
NDVI_LOG = "./ndvi_missing_log.txt"

def get_closest_ndvi(lat, lon, start_date, end_date, max_try=3):
    point = ee.Geometry.Point([lon, lat])
    collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start_date, end_date) \
        .filterBounds(point) \
        .select("NDVI") \
        .sort("system:time_start")

    image_list = collection.toList(collection.size())
    size = collection.size().getInfo()

    if size == 0:
        return None, "이미지 없음"

    for i in range(min(size, 5)): 
        try:
            image = ee.Image(image_list.get(i))
            ndvi = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=250
            ).get("NDVI").getInfo()

            if ndvi is not None:
                date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
                return ndvi * 0.0001, date
        except Exception as e:
            continue
    return None, "NDVI 값 없음 또는 예외"

def process_row(idx, row):
    lat, lon = row["latitude"], row["longitude"]
    end_date = pd.to_datetime(row["fire_end_date"])
    ndvi_start = end_date + timedelta(days=1)
    ndvi_end = ndvi_start + timedelta(days=30)

    try:
        ndvi_val, source = get_closest_ndvi(lat, lon, ndvi_start.strftime("%Y-%m-%d"), ndvi_end.strftime("%Y-%m-%d"))
        result = {
            "index": idx,
            "ndvi_after": ndvi_val,
            "ndvi_date": source if ndvi_val is not None else None,
            "ndvi_status": "✅" if ndvi_val is not None else f"❌ {source}"
        }
        print(f"[{idx}] NDVI 결과: {result}")
        return result
    except Exception as e:
        print(f"[{idx}] 예외 발생: {e}")
        return {
            "index": idx,
            "ndvi_after": None,
            "ndvi_date": None,
            "ndvi_status": f"❌ 예외: {e}"
        }

def main():
    df = pd.read_csv(INPUT_CSV)
    df["ndvi_after"] = None
    df["ndvi_date"] = None

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_row, idx, row) for idx, row in df.iterrows()]
        for future in futures:
            result = future.result()
            results.append(result)

    # 결과 병합 (index 기준 직접 삽입)
    for r in results:
        idx = r["index"]
        df.at[idx, "ndvi_after"] = r["ndvi_after"]
        df.at[idx, "ndvi_date"] = r["ndvi_date"]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ NDVI 병합 완료 → {OUTPUT_CSV}")

    # 실패 로그 저장
    with open(NDVI_LOG, "w", encoding="utf-8") as log_file:
        for r in results:
            if r["ndvi_after"] is None:
                log_file.write(f"{r['index']}: {r['ndvi_status']}\n")

if __name__ == "__main__":
    main()