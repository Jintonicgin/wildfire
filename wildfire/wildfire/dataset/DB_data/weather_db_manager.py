import cx_Oracle
import datetime
import json
import sys
import requests
import numpy as np
import traceback
import pandas as pd

# Oracle Client 초기화 (oracle_db.py와 동일한 경로 사용)
cx_Oracle.init_oracle_client(lib_dir="/Users/mmymacymac/Developer/Tools/instantclient_19_25")

def get_db_connection():
    try:
        dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")
        conn = cx_Oracle.connect(user="wildfire", password="1234", dsn=dsn)
        return conn
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 연결 오류: {e}")
        return None

def drop_all_tables():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        cursor = conn.cursor()
        
        tables_to_drop = ["GEE_FEATURES", "NASA_HOURLY_WEATHER", "NASA_DAILY_PRECIP"]
        for table_name in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                print(f"✅ 기존 {table_name} 테이블 삭제 완료.")
            except cx_Oracle.DatabaseError as e:
                if "ORA-00942" not in str(e): # ORA-00942: table or view does not exist
                    print(f"⚠️ {table_name} 테이블 삭제 중 오류 발생 (무시 가능): {e}")
        conn.commit()
        return True
    except cx_Oracle.DatabaseError as e:
        print(f"❌ 테이블 삭제 중 오류: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def create_all_tables():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        cursor = conn.cursor()
        
        # GEE_FEATURES 테이블 생성
        cursor.execute("""
            CREATE TABLE GEE_FEATURES (
                LATITUDE                NUMBER(9,6)     NOT NULL,
                LONGITUDE               NUMBER(9,6)     NOT NULL,
                NDVI_BEFORE             NUMBER(10,4),
                TREECOVER_PRE_FIRE_5X5  NUMBER(5,2),
                ELEVATION_MEAN          NUMBER(10,2),
                ELEVATION_MIN           NUMBER(10,2),
                ELEVATION_STD           NUMBER(10,2),
                SLOPE_MEAN              NUMBER(10,2),
                SLOPE_MIN               NUMBER(10,2),
                SLOPE_MAX               NUMBER(10,2),
                SLOPE_STD               NUMBER(10,2),
                ASPECT_MODE             NUMBER(10,2),
                ASPECT_STD              NUMBER(10,2),
                LAST_UPDATED            TIMESTAMP(6)    DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT PK_GEE_FEATURES PRIMARY KEY (LATITUDE, LONGITUDE)
            )
        """)
        print("✅ GEE_FEATURES 테이블 생성 완료.")

        # NASA_HOURLY_WEATHER 테이블 생성
        cursor.execute("""
            CREATE TABLE NASA_HOURLY_WEATHER (
                LATITUDE                NUMBER(9,6)     NOT NULL,
                LONGITUDE               NUMBER(9,6)     NOT NULL,
                DATA_DATE               DATE            NOT NULL,
                DATA_HOUR               NUMBER(2,0)     NOT NULL,
                T2M                     NUMBER(6,2),
                RH2M                    NUMBER(6,2),
                WS2M                    NUMBER(6,2),
                WD2M                    NUMBER(6,2),
                PRECTOTCORR             NUMBER(8,3),
                PS                      NUMBER(8,2),
                ALLSKY_SFC_SW_DWN       NUMBER(10,2),
                WS10M                   NUMBER(6,2),
                WD10M                   NUMBER(6,2),
                LAST_UPDATED            TIMESTAMP(6)    DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT PK_NASA_HOURLY_WEATHER PRIMARY KEY (LATITUDE, LONGITUDE, DATA_DATE, DATA_HOUR)
            )
        """)
        print("✅ NASA_HOURLY_WEATHER 테이블 생성 완료.")

        # NASA_DAILY_PRECIP 테이블 생성
        cursor.execute("""
            CREATE TABLE NASA_DAILY_PRECIP (
                LATITUDE                NUMBER(9,6)     NOT NULL,
                LONGITUDE               NUMBER(9,6)     NOT NULL,
                DATA_DATE               DATE            NOT NULL,
                PRECTOTCORR             NUMBER(10,3),
                LAST_UPDATED            TIMESTAMP(6)    DEFAULT SYSTIMESTAMP NOT NULL,
                CONSTRAINT PK_NASA_DAILY_PRECIP PRIMARY KEY (LATITUDE, LONGITUDE, DATA_DATE)
            )
        """)
        print("✅ NASA_DAILY_PRECIP 테이블 생성 완료.")

        conn.commit()
        return True
    except cx_Oracle.DatabaseError as e:
        print(f"❌ 테이블 생성 중 오류: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_nasa_hourly_weather(hourly_data_list):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return 0
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO NASA_HOURLY_WEATHER (
                LATITUDE, LONGITUDE, DATA_DATE, DATA_HOUR,
                T2M, RH2M, WS2M, WD2M, PRECTOTCORR, PS, ALLSKY_SFC_SW_DWN, WS10M, WD10M
            ) VALUES (
                :latitude, :longitude, :data_date, :data_hour,
                :t2m, :rh2m, :ws2m, :wd2m, :prectotcorr, :ps, :allsky_sfc_sw_dwn, :ws10m, :wd10m
            )
        """
        
        rows_inserted = 0
        for data in hourly_data_list:
            try:
                # datetime 객체를 date와 hour로 분리
                data_date = data['dt'].date()
                data_hour = data['dt'].hour

                cursor.execute(insert_sql, {
                    "latitude": data["lat"],
                    "longitude": data["lon"],
                    "data_date": data_date,
                    "data_hour": data_hour,
                    "t2m": data.get("T2M"),
                    "rh2m": data.get("RH2M"),
                    "ws2m": data.get("WS2M"),
                    "wd2m": data.get("WD2M"),
                    "prectotcorr": data.get("PRECTOTCORR"),
                    "ps": data.get("PS"),
                    "allsky_sfc_sw_dwn": data.get("ALLSKY_SFC_SW_DWN"),
                    "ws10m": data.get("WS10M"),
                    "wd10m": data.get("WD10M")
                })
                rows_inserted += 1
            except cx_Oracle.IntegrityError as e:
                if "ORA-00001" in str(e):
                    pass # 중복 데이터는 무시하고 다음으로 진행
                else:
                    print(f"❌ NASA_HOURLY_WEATHER 삽입 중 무결성 오류: {e} (데이터: {data})")
            except Exception as e:
                print(f"❌ NASA_HOURLY_WEATHER 삽입 중 오류: {e} (데이터: {data})")
        
        conn.commit()
        print(f"✅ {rows_inserted}건의 NASA_HOURLY_WEATHER 데이터 삽입 완료.")
        return rows_inserted
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 오류: {e}")
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_nasa_daily_precip(daily_data_list):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return 0
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO NASA_DAILY_PRECIP (
                LATITUDE, LONGITUDE, DATA_DATE, PRECTOTCORR
            ) VALUES (
                :latitude, :longitude, :data_date, :prectotcorr
            )
        """
        
        rows_inserted = 0
        for data in daily_data_list:
            try:
                cursor.execute(insert_sql, {
                    "latitude": data["lat"],
                    "longitude": data["lon"],
                    "data_date": data["dt"].date(),
                    "prectotcorr": data.get("PRECTOTCORR")
                })
                rows_inserted += 1
            except cx_Oracle.IntegrityError as e:
                if "ORA-00001" in str(e):
                    pass # 중복 데이터는 무시하고 다음으로 진행
                else:
                    print(f"❌ NASA_DAILY_PRECIP 삽입 중 무결성 오류: {e} (데이터: {data})")
            except Exception as e:
                print(f"❌ NASA_DAILY_PRECIP 삽입 중 오류: {e} (데이터: {data})")
        
        conn.commit()
        print(f"✅ {rows_inserted}건의 NASA_DAILY_PRECIP 데이터 삽입 완료.")
        return rows_inserted
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 오류: {e}")
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_gee_features(gee_data):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO GEE_FEATURES (
                LATITUDE, LONGITUDE, NDVI_BEFORE, TREECOVER_PRE_FIRE_5X5,
                ELEVATION_MEAN, ELEVATION_MIN, ELEVATION_STD,
                SLOPE_MEAN, SLOPE_MIN, SLOPE_MAX, SLOPE_STD,
                ASPECT_MODE, ASPECT_STD
            ) VALUES (
                :latitude, :longitude, :ndvi_before, :treecover_pre_fire_5x5,
                :elevation_mean, :elevation_min, :elevation_std,
                :slope_mean, :slope_min, :slope_max, :slope_std,
                :aspect_mode, :aspect_std
            )
        """
        
        try:
            cursor.execute(insert_sql, {
                "latitude": gee_data["lat"],
                "longitude": gee_data["lon"],
                "ndvi_before": gee_data.get("ndvi_before"),
                "treecover_pre_fire_5x5": gee_data.get("treecover_pre_fire_5x5"),
                "elevation_mean": gee_data.get("elevation_mean"),
                "elevation_min": gee_data.get("elevation_min"),
                "elevation_std": gee_data.get("elevation_std"),
                "slope_mean": gee_data.get("slope_mean"),
                "slope_min": gee_data.get("slope_min"),
                "slope_max": gee_data.get("slope_max"),
                "slope_std": gee_data.get("slope_std"),
                "aspect_mode": gee_data.get("aspect_mode"),
                "aspect_std": gee_data.get("aspect_std")
            })
            conn.commit()
            # print(f"✅ GEE_FEATURES 데이터 삽입 완료 (위도: {gee_data['lat']}, 경도: {gee_data['lon']}).")
            return True
        except cx_Oracle.IntegrityError as e:
            if "ORA-00001" in str(e):
                print(f"GEE_FEATURES 데이터 중복: 위도 {gee_data['lat']}, 경도 {gee_data['lon']} - 건너뜀")
                return False
            else:
                print(f"❌ GEE_FEATURES 삽입 중 무결성 오류: {e} (데이터: {gee_data})")
                return False
        except Exception as e:
            print(f"❌ GEE_FEATURES 삽입 중 오류: {e} (데이터: {gee_data})")
            return False
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 오류: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- 데이터 조회 함수 --- #
def get_nasa_hourly_weather_from_db(lat, lon, timestamp):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        cursor = conn.cursor()

        query = """
            SELECT T2M, RH2M, WS2M, WD2M, PRECTOTCORR, PS, ALLSKY_SFC_SW_DWN, WS10M, WD10M
            FROM NASA_HOURLY_WEATHER
            WHERE LATITUDE = :lat AND LONGITUDE = :lon
            AND DATA_DATE = :data_date AND DATA_HOUR = :data_hour
        """
        data_date = timestamp.date()
        data_hour = timestamp.hour
        cursor.execute(query, {"lat": lat, "lon": lon, "data_date": data_date, "data_hour": data_hour})
        row = cursor.fetchone()

        if row:
            return {
                "T2M": row[0],
                "RH2M": row[1],
                "WS2M": row[2],
                "WD2M": row[3],
                "PRECTOTCORR": row[4],
                "PS": row[5],
                "ALLSKY_SFC_SW_DWN": row[6],
                "WS10M": row[7],
                "WD10M": row[8],
            }
        return None
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 조회 오류 (NASA_HOURLY_WEATHER): {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_nasa_daily_precip_from_db(lat, lon, date):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        cursor = conn.cursor()

        query = """
            SELECT PRECTOTCORR
            FROM NASA_DAILY_PRECIP
            WHERE LATITUDE = :lat AND LONGITUDE = :lon
            AND DATA_DATE = :data_date
        """
        cursor.execute(query, {"lat": lat, "lon": lon, "data_date": date})
        row = cursor.fetchone()

        if row:
            return {"PRECTOTCORR": row[0]}
        return None
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 조회 오류 (NASA_DAILY_PRECIP): {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_gee_features_from_db(lat, lon):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        cursor = conn.cursor()

        query = """
            SELECT NDVI_BEFORE, TREECOVER_PRE_FIRE_5X5, ELEVATION_MEAN, ELEVATION_MIN, ELEVATION_STD,
                   SLOPE_MEAN, SLOPE_MIN, SLOPE_MAX, SLOPE_STD, ASPECT_MODE, ASPECT_STD
            FROM GEE_FEATURES
            WHERE LATITUDE = :lat AND LONGITUDE = :lon
        """
        cursor.execute(query, {"lat": lat, "lon": lon})
        row = cursor.fetchone()

        if row:
            return {
                "ndvi_before": row[0],
                "treecover_pre_fire_5x5": row[1],
                "elevation_mean": row[2],
                "elevation_min": row[3],
                "elevation_std": row[4],
                "slope_mean": row[5],
                "slope_min": row[6],
                "slope_max": row[7],
                "slope_std": row[8],
                "aspect_mode": row[9],
                "aspect_std": row[10],
            }
        return None
    except cx_Oracle.DatabaseError as e:
        print(f"❌ DB 조회 오류 (GEE_FEATURES): {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# fetch_all_weather.py의 fetch_nasa_hourly_weather 함수를 복사하여 사용
def fetch_nasa_hourly_weather_from_api(lat, lng, yyyymmdd, hour_str, max_retry=3):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=T2M,RH2M,WS2M,WD2M,PRECTOTCORR,PS,ALLSKY_SFC_SW_DWN,WS10M,WD10M"
        f"&community=RE&longitude={lng}&latitude={lat}&start={yyyymmdd}&end={yyyymmdd}&format=JSON"
    )

    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            raw_data = res.json()
            data = raw_data.get("properties", {}).get("parameter", {})

            hour_key = f"{yyyymmdd}{hour_str.zfill(2)}"

            result = {
                "T2M": data.get("T2M", {}).get(hour_key, np.nan),
                "RH2M": data.get("RH2M", {}).get(hour_key, np.nan),
                "WS2M": data.get("WS2M", {}).get(hour_key, np.nan),
                "WD2M": data.get("WD2M", {}).get(hour_key, np.nan),
                "WS10M": data.get("WS10M", {}).get(hour_key, np.nan),
                "WD10M": data.get("WD10M", {}).get(hour_key, np.nan),
                "PRECTOTCORR": data.get("PRECTOTCORR", {}).get(hour_key, np.nan),
                "PS": data.get("PS", {}).get(hour_key, np.nan),
                "ALLSKY_SFC_SW_DWN": data.get("ALLSKY_SFC_SW_DWN", {}).get(hour_key, np.nan)
            }
            return result
        except Exception as e:
            if attempt == max_retry - 1:
                raise e

def fetch_nasa_daily_precip_from_api(lat, lng, start_date, end_date, max_retry=3):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR&community=RE&longitude={lng}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )
    for attempt in range(max_retry):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            data = res.json().get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
            return data
        except Exception as e:
            if attempt == max_retry - 1:
                raise e

# predict.py의 get_gee_features 함수를 복사하여 사용
import ee

def get_gee_features_from_api(lat, lon):
    # GEE 초기화 (여기서는 매번 초기화하지 않고, 필요시 외부에서 초기화한다고 가정)
    # ee.Initialize(project='wildfire-464907') # 이 함수 호출 전에 초기화 필요
    try:
        ee.Initialize(project='wildfire-464907')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='wildfire-464907')

    point = ee.Geometry.Point([lon, lat])
    today = datetime.date.today()
    end = today - datetime.timedelta(days=5)
    start = end - datetime.timedelta(days=30)

    ndvi = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterBounds(point) \
        .filterDate(str(start), str(end)) \
        .sort('system:time_start', False) \
        .first() \
        .select('NDVI') \
        .reduceRegion(ee.Reducer.mean(), point, 250) \
        .getInfo()

    treecover = ee.Image("UMD/hansen/global_forest_change_2022_v1_10") \
        .select("treecover2000") \
        .reduceRegion(ee.Reducer.mean(), point, 30) \
        .getInfo()

    elev_img = ee.Image("USGS/SRTMGL1_003").clip(point.buffer(500))
    slope_img = ee.Terrain.slope(elev_img)
    aspect_img = ee.Terrain.aspect(elev_img)

    elev_stats = elev_img.reduceRegion(ee.Reducer.minMax().combine(
        reducer2=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(), sharedInputs=True), sharedInputs=True),
        point, 90).getInfo()

    slope_stats = slope_img.reduceRegion(ee.Reducer.minMax().combine(
        reducer2=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(), sharedInputs=True), sharedInputs=True),
        point, 90).getInfo()

    aspect_stats = aspect_img.reduceRegion(ee.Reducer.mode().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True),
        point, 90).getInfo()

    return {
        "lat": lat,
        "lon": lon,
        "ndvi_before": float(ndvi.get("NDVI", -999)) / 10000 if ndvi.get("NDVI") else -999,
        "treecover_pre_fire_5x5": float(treecover.get("treecover2000", -999)),
        "elevation_mean": float(elev_stats.get("elevation_mean", -999)),
        "elevation_min": float(elev_stats.get("elevation_min", -999)),
        "elevation_std": float(elev_stats.get("elevation_stdDev", -999)),
        "slope_mean": float(slope_stats.get("slope_mean", -999)),
        "slope_min": float(slope_stats.get("slope_min", -999)),
        "slope_max": float(slope_stats.get("slope_max", -999)),
        "slope_std": float(slope_stats.get("slope_stdDev", -999)),
        "aspect_mode": float(aspect_stats.get("aspect_mode", -999)),
        "aspect_std": float(aspect_stats.get("aspect_stdDev", -999)),
    }

# 테스트를 위한 메인 함수
if __name__ == "__main__":
    sample_lat = 37.8853  # 강원도 속초시 위도
    sample_lon = 128.5567 # 강원도 속초시 경도
    
    # 1. 기존 테이블 삭제 및 새 테이블 생성
    print("기존 테이블 삭제 시도...")
    drop_all_tables()
    print("\n새 테이블 생성 시도...")
    create_all_tables()

    # 2. 샘플 GEE Features 데이터 삽입
    print("\n샘플 GEE Features 데이터 수집 및 삽입 시도...")
    try:
        gee_features = get_gee_features_from_api(sample_lat, sample_lon)
        insert_gee_features(gee_features)
    except Exception as e:
        print(f"❌ GEE Features 데이터 수집/삽입 실패: {e}")
        print(traceback.format_exc())

    # 3. 샘플 NASA Hourly Weather 데이터 삽입
    print("\n샘플 NASA Hourly Weather 데이터 수집 및 삽입 시도...")
    today = datetime.datetime.now()
    hourly_data_to_insert = []
    for i in range(3): # 3일치 데이터
        target_date = today - datetime.timedelta(days=i)
        yyyymmdd = target_date.strftime("%Y%m%d")
        for hour in range(24): # 24시간 데이터
            hour_str = str(hour)
            try:
                hourly_data = fetch_nasa_hourly_weather_from_api(sample_lat, sample_lon, yyyymmdd, hour_str)
                if hourly_data:
                    hourly_data['dt'] = datetime.datetime.strptime(f"{yyyymmdd}{hour_str.zfill(2)}", "%Y%m%d%H")
                    hourly_data['lat'] = sample_lat
                    hourly_data['lon'] = sample_lon
                    hourly_data_to_insert.append(hourly_data)
            except Exception as e:
                print(f"❌ {yyyymmdd} {hour_str}시 NASA Hourly Weather 데이터 수집 실패: {e}")
    
    if hourly_data_to_insert:
        insert_nasa_hourly_weather(hourly_data_to_insert)
    else:
        print("수집할 샘플 NASA Hourly Weather 데이터가 없습니다.")

    # 4. 샘플 NASA Daily Precip 데이터 삽입
    print("\n샘플 NASA Daily Precip 데이터 수집 및 삽입 시도...")
    daily_data_to_insert = []
    for i in range(3): # 3일치 데이터
        target_date = today - datetime.timedelta(days=i)
        yyyymmdd = target_date.strftime("%Y%m%d")
        start_date_str = (target_date - datetime.timedelta(days=1)).strftime("%Y%m%d") # 전날부터 오늘까지
        end_date_str = yyyymmdd
        try:
            daily_precip_dict = fetch_nasa_daily_precip_from_api(sample_lat, sample_lon, start_date_str, end_date_str)
            if daily_precip_dict:
                # API 응답에서 해당 날짜의 강수량만 추출
                precip_val = daily_precip_dict.get(yyyymmdd, np.nan)
                if not np.isnan(precip_val):
                    daily_data_to_insert.append({
                        "lat": sample_lat,
                        "lon": sample_lon,
                        "dt": target_date,
                        "PRECTOTCORR": precip_val
                    })
        except Exception as e:
            print(f"❌ {yyyymmdd} NASA Daily Precip 데이터 수집 실패: {e}")

    if daily_data_to_insert:
        insert_nasa_daily_precip(daily_data_to_insert)
    else:
        print("수집할 샘플 NASA Daily Precip 데이터가 없습니다.")

    print("\n데이터베이스 작업 완료.")
