# -*- coding: utf-8 -*-
import argparse, json, os, shutil, subprocess, sys, tempfile
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from ultralytics import YOLO

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_ffprobe_fps(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=r_frame_rate","-of","default=nw=1:nk=1", path],
            stderr=subprocess.STDOUT,
        ).decode("utf-8","ignore").strip()
        if not out: return None
        if "/" in out:
            num, den = out.split("/", 1)
            num = float(num); den = float(den)
            return (num / den) if den != 0 else None
        return float(out)
    except Exception:
        return None

def find_video_in_dir(d: str) -> Optional[str]:
    best = None
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXTS):
                full = os.path.join(root, fn)
                try: sz = os.path.getsize(full)
                except Exception: sz = 0
                if best is None or sz > best[0]:
                    best = (sz, full)
    return best[1] if best else None

def auto_select_device(user_device: str = "") -> str:
    if user_device: return user_device
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def build_classes_arg(model: YOLO, targets_csv: str) -> Optional[List[int]]:
    targets = [t.strip() for t in (targets_csv or "").split(",") if t.strip()]
    if not targets: return None
    names = getattr(model, "names", None)
    if not names: return None
    if isinstance(names, dict):
        name_to_id = {str(v): int(k) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        name_to_id = {str(v): i for i, v in enumerate(names)}
    else:
        return None
    ids = [name_to_id[t] for t in targets if t in name_to_id]
    return ids or None

def label_from_names(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)):
        return str(names[cls_id]) if 0 <= cls_id < len(names) else str(cls_id)
    return str(cls_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="YOLO .pt weights")
    ap.add_argument("--video-in", required=True, help="input video path")
    ap.add_argument("--video-out", required=True, help="annotated output video path")
    ap.add_argument("--conf", type=float, default=0.05, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    ap.add_argument("--targets", type=str, default="fire,smoke", help="comma-separated labels to keep")
    ap.add_argument("--save-jsonl", required=True, help="detections jsonl path")
    ap.add_argument("--summary", required=True, help="summary json path")
    # inference params
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--agnostic-nms", action="store_true")
    ap.add_argument("--max-det", type=int, default=2000)
    ap.add_argument("--device", type=str, default="", help="cuda|mps|cpu (auto if empty)")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation")

    # (웹 호환을 위해 받기만 하고 무시하는 옵션들)
    ap.add_argument("--gate-enable", action="store_true")
    ap.add_argument("--gate-min-frac", type=float, default=0.10)
    ap.add_argument("--gate-debug", action="store_true")
    ap.add_argument("--gate-use-raft", action="store_true")
    ap.add_argument("--min-area-frac", type=float, default=0.0)
    ap.add_argument("--focus-label", type=str, default="")
    ap.add_argument("--gate-smoke-min-frac", type=float, default=0.03)
    ap.add_argument("--gate-smoke-use-motion", action="store_true")
    ap.add_argument("--gate-smoke-motion-min-frac", type=float, default=0.02)
    ap.add_argument("--motion-diff-thr", type=float, default=0.06)
    ap.add_argument("--gate-downsample", type=int, default=1)
    ap.add_argument("--gate-every", type=int, default=1)
    ap.add_argument("--bigfire-thr", type=float, default=0.0)

    ap.add_argument("--tracking", type=str, default="byte", choices=["off", "byte"],
                    help="Use ByteTrack to persist boxes across frames")
    ap.add_argument("--track-buffer", type=int, default=60, help="frames to keep lost tracks")
    ap.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack match threshold")

    args = ap.parse_args()

    in_path = os.path.abspath(args.video_in)
    out_path = os.path.abspath(args.video_out)
    ensure_dir(os.path.dirname(out_path))
    ensure_dir(os.path.dirname(os.path.abspath(args.save_jsonl)))
    ensure_dir(os.path.dirname(os.path.abspath(args.summary)))

    tmpdir = tempfile.mkdtemp(prefix="yolo_pred_")
    device = auto_select_device(args.device)
    print(f"[INFO] Using device: {device}", file=sys.stderr)

    # model
    model = YOLO(args.model)

    # class filter
    classes_arg = build_classes_arg(model, args.targets)
    if classes_arg is not None:
        print(f"[INFO] Filtering classes by targets: {args.targets} -> {classes_arg}", file=sys.stderr)
    else:
        print(f"[INFO] No class filter applied (targets='{args.targets}')", file=sys.stderr)

    # fps
    fps = run_ffprobe_fps(in_path)
    if fps:
        print(f"[INFO] Input FPS: {fps:.3f}", file=sys.stderr)
    else:
        print("[WARN] Could not determine FPS; 't' will be frame index.", file=sys.stderr)

    # predict stream (Ultralytics가 주석 비디오도 tmpdir에 생성)
    try:
        results = model.predict(
            source=in_path,
            save=True,
            save_txt=False,
            save_conf=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            vid_stride=args.vid_stride,
            agnostic_nms=False,
            max_det=args.max_det,
            project=tmpdir,
            name="",
            exist_ok=True,
            device=device,
            stream=True,
            classes=classes_arg,
            augment=args.tta,
        )
    except Exception:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        raise

    jf = open(args.save_jsonl, "w", encoding="utf-8")
    first_time: Dict[str, float] = {}
    max_conf: Dict[str, float] = {}
    any_detection = False

    for frame_idx, r in enumerate(results):
        t_val: Optional[float] = (frame_idx / fps) if (fps and fps > 0) else None

        frame_dets: List[Dict[str, Any]] = []
        if getattr(r, "boxes", None) is not None and r.boxes is not None:
            cls_tensor = r.boxes.cls
            conf_tensor = r.boxes.conf
            names = getattr(model, "names", {})

            target_set = {t.strip() for t in (args.targets or "").split(",") if t.strip()}

            for i in range(len(cls_tensor)):
                cls_id = int(cls_tensor[i].item())
                label = label_from_names(names, cls_id)
                conf = float(conf_tensor[i].item())

                if target_set and label not in target_set:
                    continue

                one = {
                    "t": t_val if t_val is not None else frame_idx,
                    "label": label,
                    "conf": conf,
                }
                frame_dets.append(one)

                if t_val is not None and label not in first_time:
                    first_time[label] = t_val
                max_conf[label] = max(conf, max_conf.get(label, 0.0))

        if frame_dets:
            any_detection = True
            jf.write(json.dumps({"detections": frame_dets}, ensure_ascii=False) + "\n")

    jf.close()

    # summary
    events = []
    labels = sorted(set(list(first_time.keys()) + list(max_conf.keys())))
    for label in labels:
        events.append({
            "label": label,
            "first_time": first_time.get(label, None),
            "max_conf": max_conf.get(label, None),
        })
    with open(args.summary, "w", encoding="utf-8") as fsum:
        json.dump({"events": events, "fps": fps}, fsum, ensure_ascii=False, indent=2)

    # move annotated video out
    pred_video = find_video_in_dir(tmpdir)
    if pred_video is None:
        print("[WARN] No annotated video found in:", tmpdir, file=sys.stderr)
    else:
        try:
            if os.path.isfile(out_path):
                os.remove(out_path)
        except Exception:
            pass
        shutil.move(pred_video, out_path)

    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    if any_detection:
        print("[INFO] Detections found and saved.", file=sys.stderr)
    else:
        print("[INFO] No detections.", file=sys.stderr)

if __name__ == "__main__":
    main()