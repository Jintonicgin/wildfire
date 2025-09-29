import os, glob, math

roots = [
    "wildfire/ML/wildfire_yolo/train/labels",
    "wildfire/ML/wildfire_yolo/valid/labels",
    "wildfire/ML/wildfire_yolo/test/labels",
]

fix_cnt = bad_cnt = del_cnt = 0
bad_samples = []

def ok01(x):
    return (x is not None) and (not math.isnan(x)) and (not math.isinf(x)) and (0.0 <= x <= 1.0)

for root in roots:
    if not os.path.isdir(root):
        continue
    for p in glob.glob(os.path.join(root, "*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        out = []
        changed = False
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            # YOLO: cls cx cy w h  → 정확히 5개
            if len(parts) != 5:
                bad_cnt += 1
                bad_samples.append((p, i, "COLS", ln))
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                bad_cnt += 1
                bad_samples.append((p, i, "PARSE", ln))
                continue

            # 클래스 0/1만 허용
            if c not in (0, 1):
                bad_cnt += 1
                bad_samples.append((p, i, f"CLS_{c}", ln))
                continue

            # 범위 체크
            if not (ok01(x) and ok01(y) and w > 0.0 and h > 0.0 and w <= 1.0 and h <= 1.0):
                bad_cnt += 1
                bad_samples.append((p, i, "RANGE", ln))
                continue

            out.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        if out:
            if out != lines:
                with open(p, "w", encoding="utf-8") as f:
                    f.write("\n".join(out) + "\n")
                fix_cnt += 1
        else:
            # 유효 라인이 0이면 해당 라벨 파일 삭제(=배경 이미지로 처리)
            os.remove(p)
            del_cnt += 1

print(f"[label-fix] rewritten={fix_cnt}, removed_empty={del_cnt}, bad_lines_skipped={bad_cnt}")
print("[label-fix] sample bad (first 20):")
for s in bad_samples[:20]:
    print(" -", s[0], "line", s[1], s[2], "→", s[3])