# probe_loader.py
from ultralytics import YOLO

def on_train_batch_start(trainer):
    import torch, traceback
    try:
        # 아직 로더가 배치를 전달하기 전이면 스킵
        if not hasattr(trainer, "batch") or trainer.batch is None:
            return

        b  = trainer.batch
        bs = int(trainer.batch_size)

        # 경로
        paths = b.get("im_file") or b.get("path") or ()
        if isinstance(paths, str):
            paths = (paths,)
        print("\n[probe] path0:", paths[0] if paths else "(unknown)")

        # 기본 텐서
        bi = b.get("batch_idx")   # (N,)
        cc = b.get("cls")         # (N,1)
        bb = b.get("bboxes")      # (N,4)

        shp = lambda x: None if x is None else tuple(x.shape)
        print(f"[probe] shapes  batch_idx={shp(bi)}  cls={shp(cc)}  bboxes={shp(bb)}")

        # --- batch_idx 점검 ---
        if bi is None:
            print("[probe][FATAL] batch_idx is None"); raise SystemExit(1)
        if not torch.all(torch.isfinite(bi)):
            print("[probe][FATAL] batch_idx has NaN/Inf"); raise SystemExit(1)

        # 정수형 보장(혹시 float로 올 수 있어도 bincount 위해 int64로)
        bi_i64 = bi.to(torch.int64)
        bi_min = int(bi_i64.min()) if bi_i64.numel() else 0
        bi_max = int(bi_i64.max()) if bi_i64.numel() else -1
        print(f"[probe] batch_idx range = [{bi_min}, {bi_max}] (must be in [0, {bs-1}])")

        if bi_min < 0 or bi_max >= bs:
            bad = bi_i64[(bi_i64 < 0) | (bi_i64 >= bs)]
            print("[probe][FATAL] out-of-range batch_idx (sample):",
                  bad[:10].detach().cpu().tolist())
            print("[probe] offending paths:", paths)
            raise SystemExit(1)

        # --- 라벨 값 점검 ---
        if cc is not None and cc.numel():
            if not torch.all(torch.isfinite(cc)):
                print("[probe][FATAL] cls NaN/Inf"); raise SystemExit(1)
            if cc.min() < 0:
                print("[probe][FATAL] negative class id"); raise SystemExit(1)
        if bb is not None and bb.numel():
            if not torch.all(torch.isfinite(bb)):
                print("[probe][FATAL] bboxes NaN/Inf"); raise SystemExit(1)
            mn, mx = bb.min(0).values, bb.max(0).values
            print(f"[probe] xywh min={mn.tolist()}  max={mx.tolist()}")
            if (mn[0] < 0) or (mn[1] < 0) or (mn[2] <= 0) or (mn[3] <= 0) or \
               (mx[0] > 1) or (mx[1] > 1) or (mx[2] > 1) or (mx[3] > 1):
                print("[probe][FATAL] bboxes out of [0,1] or w/h<=0"); raise SystemExit(1)

        # --- 이 배치의 이미지당 GT 개수와 최대값 ---
        # (ultralytics 내부와 동일하게 bincount로 확인)
        counts = torch.bincount(bi_i64, minlength=bs)
        cmax = int(counts.max())
        print(f"[probe] per-image counts = {counts.tolist()}  (max={cmax})")

        # --- 이미지 텐서 dtype 정리 (MPSByteType → float32/255) ---
        img = b.get("img", None)
        if img is not None:
            if img.dtype != torch.float32:
                img = img.float()
            # 장치로 옮기고 정규화
            b["img"] = img.div_(255.0).to(trainer.device, non_blocking=True).contiguous()

    except SystemExit:
        print("[probe] STOP to fix labels/indices.")
        raise
    except Exception:
        print("[probe][UNEXPECTED]; paths:", paths)
        traceback.print_exc()
        raise SystemExit(1)


def main():
    model = YOLO("yolov8m.pt")
    model.add_callback("on_train_batch_start", on_train_batch_start)

    # 캐시가 라벨 수정 이전 상태일 수 있으니 먼저 지우기 권장
    # ! rm -f wildfire/ML/wildfire_yolo/train/labels.cache wildfire/ML/wildfire_yolo/valid/labels.cache

    model.train(
        data="wildfire/ML/wildfire_yolo/data.yaml",
        imgsz=640, batch=1, epochs=1,
        device="mps", workers=0,
        mosaic=0.0, copy_paste=0.0, scale=0.0, translate=0.0,
        val=False, save=False, name="_probe_loader422"
    )

if __name__ == "__main__":
    main()