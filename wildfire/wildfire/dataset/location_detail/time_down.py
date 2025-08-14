import requests
import xml.etree.ElementTree as ET
import pandas as pd

SERVICE_KEY = "1l+jrCYzmud0jp/CrmoEltsVKql02sltWLgS2oGG0sBx6gllyYCJCIFb5R1J0wbuN/+xUFcg4h4rXKsMFe63fg=="
BASE_URL = "http://apis.data.go.kr/1400000/forestStusService/getfirestatsservice"

def fetch_fire_data_by_year(year):
    params = {
        "serviceKey": SERVICE_KEY,
        "searchStDt": f"{year}0101",
        "searchEdDt": f"{year}1231",
        "pageNo": "1",
        "numOfRows": "1000",
        "type": "xml"
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.encoding = "utf-8"
        if response.status_code != 200:
            print(f"❌ {year} 요청 실패: HTTP {response.status_code}")
            return []
        root = ET.fromstring(response.content)
    except Exception as e:
        print(f"❌ {year} 요청 오류: {e}")
        return []

    records = []
    for item in root.findall(".//item"):
        locsi = item.findtext("locsi", "").strip()
        if locsi != "강원":
            continue  # 강원도 데이터만 필터링

        # 산불 시작 연월일시
        startyear = item.findtext("startyear", "").strip()
        startmonth = item.findtext("startmonth", "").zfill(2)
        startday = item.findtext("startday", "").zfill(2)
        starttime = item.findtext("starttime", "").strip()
        fire_start = f"{startyear}-{startmonth}-{startday}"

        # 진화 종료 연월일시
        endyear = item.findtext("endyear", "").strip()
        endmonth = item.findtext("endmonth", "").zfill(2)
        endday = item.findtext("endday", "").zfill(2)
        endtime = item.findtext("endtime", "").strip()
        fire_end = f"{endyear}-{endmonth}-{endday}"

        locgungu = item.findtext("locgungu", "").strip()  # 시군구
        locmenu = item.findtext("locmenu", "").strip()    # 읍면동
        damage_area_str = item.findtext("damagearea", "0").strip()  # 피해면적
        try:
            damage_area = float(damage_area_str)
        except:
            damage_area = 0.0

        records.append({
            "fire_start_datetime": fire_start,
            "start_time": starttime,
            "fire_end_datetime": fire_end,
            "end_time": endtime,
            "locgungu": locgungu,
            "locmenu": locmenu,
            "damage_area": damage_area
        })

    return records

if __name__ == "__main__":
    all_data = []
    for year in range(2011, 2025):
        print(f"데이터 수집 중: {year}년")
        data = fetch_fire_data_by_year(year)
        all_data.extend(data)

    # DataFrame으로 변환
    df = pd.DataFrame(all_data)

    # 결과 확인
    print(df.head())

    # CSV 저장
    df.to_csv("gangwon_fire_fires.csv", index=False, encoding="utf-8-sig")
    print("✅ 강원도 산불 데이터 CSV 저장 완료")