import pandas as pd
import requests
import time

KAKAO_REST_API_KEY = "ba322bcc9ee477ac1f3fac7a5c294fe6"

# 엑셀 혹은 CSV 경로 지정 (필요한 경로로 수정)
INPUT_FILE_PATH = "./location_detail/gangwon_fire_data.xlsx"
OUTPUT_CSV_PATH = "./location_detail/gangwon_fire_data_with_coords.csv"

def get_coordinates(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        documents = data.get("documents")
        if documents:
            first = documents[0]
            return float(first["y"]), float(first["x"])
        else:
            print(f"⚠️ 주소 검색 결과 없음: {address}")
            return None, None
    except Exception as e:
        print(f"❌ API 요청 실패: {e} - 주소: {address}")
        return None, None

def main():
    # 엑셀 파일 읽기 (CSV면 pd.read_csv 사용)
    df = pd.read_excel(INPUT_FILE_PATH, engine='openpyxl')

    latitudes = []
    longitudes = []

    for idx, row in df.iterrows():
        address = row['corrected_address']
        print(f"Processing {idx}: {address}")
        lat, lon = get_coordinates(address)
        latitudes.append(lat)
        longitudes.append(lon)
        time.sleep(0.2)  # 카카오 API 요청 제한 고려 (5 TPS)

    df['latitude'] = latitudes
    df['longitude'] = longitudes

    # 위경도 포함한 CSV로 저장
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 위경도 추가 후 CSV 저장 완료: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()