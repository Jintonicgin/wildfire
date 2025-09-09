import pandas as pd
import numpy as np

# 파일 경로
INPUT_CSV = "./gangwon_fire_data_full_merged.csv"
OUTPUT_CSV = "./gangwon_fire_data_with_direction.csv"

# ⬇️ 바람 벡터 → 방향 각도 계산 (0~360도, 북 = 0도)
def calculate_angle(x, y):
    angle = np.degrees(np.arctan2(x, y))
    return (angle + 360) % 360

# ⬇️ 방향 각도 → 8방위 변환
def convert_to_8_directions(angle):
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return directions[int(((angle + 22.5) % 360) // 45)]

# ⬇️ 데이터 불러오기
df = pd.read_csv(INPUT_CSV)

# ⬇️ spread_direction_label, spread_direction_class_8 계산 및 콘솔 출력
spread_angles = []
spread_classes = []

for i, row in df.iterrows():
    x = row.get("wind_x_10m")
    y = row.get("wind_y_10m")

    if pd.notnull(x) and pd.notnull(y):
        angle = calculate_angle(x, y)
        direction = convert_to_8_directions(angle)
        print(f"[{i}] wind_x_10m={x:.2f}, wind_y_10m={y:.2f} → angle={angle:.1f}°, class={direction}")
    else:
        angle = np.nan
        direction = np.nan
        print(f"[{i}] wind_x_10m or wind_y_10m 결측치 → 계산 불가")

    spread_angles.append(angle)
    spread_classes.append(direction)

# ⬇️ 컬럼 추가 및 저장
df["spread_direction_label"] = spread_angles
df["spread_direction_class_8"] = spread_classes

df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n✅ spread_direction 계산 및 저장 완료 → {OUTPUT_CSV}")