import requests
import xml.etree.ElementTree as ET
import cx_Oracle
cx_Oracle.init_oracle_client(lib_dir="/Users/mmymacymac/Developer/Tools/instantclient_19_25")

def fetch_gangwon_fire_data_by_year(year):
    url = "http://apis.data.go.kr/1400000/forestStusService/getfirestatsservice"
    service_key = "Zq6C1pNUKb7fdhRQBlkBie77nX/B+2jn2LBQo8MbKgwk0yLXvze6DVeJdPF8h6w0wkyk8dIiu8MEZEY5ioVSfw=="

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
    all_fires = []
    for year in range(2011, 2025):
        yearly_fires = fetch_gangwon_fire_data_by_year(year)
        print(f"📅 {year}년: {len(yearly_fires)}건 수집됨")
        all_fires.extend(yearly_fires)

    print(f"🔥 전체 수집 건수: {len(all_fires)}건")
    insert_to_wildfire_recovery(all_fires)