import requests
import xml.etree.ElementTree as ET
import cx_Oracle
import json
import numpy as np # numpy types might be in features

# Oracle Instant Client 경로 설정 (사용자 환경에 맞게 수정 필요)
cx_Oracle.init_oracle_client(lib_dir="/Users/mmymacymac/Developer/Tools/instantclient_19_25")

class OracleDB:
    def __init__(self):
        self.conn = None
        self.cursor = None
        try:
            # 데이터베이스 연결 정보 (사용자 환경에 맞게 수정 필요)
            dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")
            self.conn = cx_Oracle.connect(user="wildfire", password="1234", dsn=dsn)
            self.cursor = self.conn.cursor()
            print("✅ Oracle DB 연결 성공.")
        except cx_Oracle.DatabaseError as e:
            print(f"❌ Oracle DB 연결 실패: {e}")
            self.conn = None
            self.cursor = None

    def insert_mountain_features(self, features):
        if not self.conn:
            print("❌ DB 연결이 유효하지 않아 데이터를 삽입할 수 없습니다.")
            return

        # 테이블 컬럼과 features 딕셔너리 키 매핑
        # SQL 컬럼명은 대문자, Python features 키는 소문자
        # features 딕셔너리에서 직접 가져올 수 있는 키만 매핑
        column_mapping = {
            "region_name": "REGION_NAME", # REGION_NAME 컬럼 추가
            "lat": "LAT", "lng": "LNG",
            "T2M": "T2M", "RH2M": "RH2M", "WS10M": "WS10M", "WD10M": "WD10M",
            "PRECTOTCORR": "PRECTOTCORR", "PS": "PS", "ALLSKY_SFC_SW_DWN": "ALLSKY_SFC_SW_DWN",
            "elevation_mean": "ELEVATION_MEAN", "elevation_min": "ELEVATION_MIN",
            "elevation_max": "ELEVATION_MAX", "elevation_std": "ELEVATION_STD",
            "slope_mean": "SLOPE_MEAN", "slope_min": "SLOPE_MIN",
            "slope_max": "SLOPE_MAX", "slope_std": "SLOPE_STD",
            "aspect_mode": "ASPECT_MODE", "aspect_std": "ASPECT_STD",
            "ndvi_before": "NDVI_BEFORE", "treecover_pre_fire_5x5": "TREECOVER_PRE_FIRE_5X5",
            "FFMC": "FFMC", "DMC": "DMC", "DC": "DC", "ISI": "ISI", "BUI": "BUI", "FWI": "FWI",
            "dry_windy_combo": "DRY_WINDY_COMBO", "fuel_combo": "FUEL_COMBO",
            "potential_spread_index": "POTENTIAL_SPREAD_INDEX", "terrain_var_effect": "TERRAIN_VAR_EFFECT",
            "wind_steady_flag": "WIND_STEADY_FLAG", "dry_to_rain_ratio_30d": "DRY_TO_RAIN_RATIO_30D",
            "ndvi_stress": "NDVI_STRESS",
            "is_spring": "IS_SPRING", "is_summer": "IS_SUMMER", "is_autumn": "IS_AUTUMN", "is_winter": "IS_WINTER"
        }

        columns = []
        bind_vars = []
        placeholders = []

        for py_key, sql_col in column_mapping.items():
            if py_key in features:
                columns.append(sql_col)
                placeholders.append(f":{len(placeholders) + 1}")
                
                value = features[py_key]
                # Handle numpy types and boolean for Oracle compatibility
                if isinstance(value, (np.float32, np.float64)):
                    bind_vars.append(float(value))
                elif isinstance(value, bool):
                    bind_vars.append(int(value))
                else:
                    bind_vars.append(value)
            else:
                print(f"⚠️ Warning: Feature '{py_key}' not found in input data for column '{sql_col}'. Skipping.")

        insert_sql = f"INSERT INTO REGION_PREDICTION_FEATURES ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        try:
            self.cursor.execute(insert_sql, bind_vars)
            self.conn.commit()
            print(f"✅ 피처 데이터 (LAT: {features.get('lat')}, LNG: {features.get('lng')}) REGION_PREDICTION_FEATURES 테이블에 삽입 완료.")
        except cx_Oracle.DatabaseError as e:
            print(f"❌ REGION_PREDICTION_FEATURES 테이블 삽입 오류: {e}")
            print(f"SQL: {insert_sql}")
            print(f"Bind Variables: {bind_vars}")
            self.conn.rollback() # 오류 발생 시 롤백

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✅ Oracle DB 연결 해제.")

# --- 기존 함수들 (OracleDB 클래스와는 별개) ---
def fetch_gangwon_fire_data_by_year(year):
    url = "http://apis.data.go.kr/1400000/forestStusService/getfirestatsservice"
    service_key = "Zq6C1pNUKb7fdhRQBlkBie77nX/B+2jn2LBQo8MbKgwk0yLXvze6DVeJdPF8h6w0wkyk8dIiu8MEZEY5ioVSfw==" # 실제 서비스 키로 대체 필요

    start_ymd = f"{year}0101"
    end_ymd = f"{year}1231"

    params = {
        "serviceKey": service_key,
        "searchStDt": start_ymd,
        "searchEdDt": end_ymd,
        "pageNo": "1",
        "numOfRows": "1000",
        "type": "xml"
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        res.encoding = 'utf-8'
        if res.status_code != 200:
            print(f"❌ {year} 요청 실패 (status {res.status_code})")
            return []
        root = ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ {year} 요청 중 오류 발생: {e}")
        return []

    fire_list = []
    for item in root.findall(".//item"):
        locsi = item.findtext("locsi", "").strip()
        if locsi != "강원":
            continue

        year = item.findtext("startyear", "")
        month = item.findtext("startmonth", "").zfill(2)
        day = item.findtext("startday", "").zfill(2)
        fire_date = f"{year}-{month}-{day}"

        gungu = item.findtext("locgungu", "").strip()
        address = f"강원도 {gungu}".strip()

        area_str = item.findtext("damagearea", "0").strip()
        try:
            area = float(area_str)
        except ValueError:
            area = 0.0

        fire_list.append({
            "fire_date": fire_date,
            "location_name": address,
            "fire_area": area
        })

    return fire_list

def insert_to_wildfire_recovery(fire_list):
    conn = None
    cursor = None
    try:
        dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")
        conn = cx_Oracle.connect(user="wildfire", password="wildfire1234", dsn=dsn)
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO wildfire_recovery (
                id, fire_date, location_name, fire_area
            ) VALUES (
                wildfire_recovery_seq.NEXTVAL, TO_DATE(:1, 'YYYY-MM-DD'), :2, :3
            )
        """

        for fire in fire_list:
            cursor.execute(insert_sql, (
                fire["fire_date"],
                fire["location_name"],
                fire["fire_area"]
            ))

        conn.commit()
        print(f"✅ {len(fire_list)}건의 데이터를 wildfire_recovery 테이블에 삽입 완료.")
    except cx_Oracle.DatabaseError as e:
        print("❌ DB 오류:", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    # 이 부분은 기존 산불 데이터 수집 및 삽입 로직입니다.
    # MOUNTAIN_FEATURES 삽입과는 별개로 동작합니다.
    all_fires = []
    for year in range(2011, 2025):
        yearly_fires = fetch_gangwon_fire_data_by_year(year)
        print(f"📅 {year}년: {len(yearly_fires)}건 수집됨")
        all_fires.extend(yearly_fires)

    print(f"🔥 전체 수집 건수: {len(all_fires)}건")
    insert_to_wildfire_recovery(all_fires)