# -*- coding: utf-8 -*-
import os, sys, json, uuid, subprocess, platform
from datetime import datetime
from typing import List, Dict, Any
from flask import Blueprint, request, jsonify, current_app, render_template, url_for
from werkzeug.utils import secure_filename

bp = Blueprint("vision", __name__, url_prefix="/vision")

ALLOWED_EXTS = {"mp4", "mov", "m4v", "avi", "mkv", "webm"}

MODEL_FIXED = "wildfire/ML/weights/best.pt"
IOU_FIXED = 0.65
CONF_CANDIDATES = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]  # 위부터 순차 시도
TARGETS = "fire,smoke"

AGNOSTIC_NMS_DEFAULT = False

IMGSZ_DEFAULT = 1280
VID_STRIDE_DEFAULT = 1
MAX_DET_DEFAULT = 2000
DEVICE_FIXED = os.environ.get("WILDFIRE_DEVICE", "mps")

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _is_allowed(filename: str) -> bool:
    return bool(filename and "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTS)

def _static_path(*parts: str) -> str:
    return os.path.join(current_app.static_folder, *parts)

def _static_url(rel_path: str) -> str:
    return url_for("static", filename=rel_path)

def _read_detections_head(jsonl_path: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(jsonl_path): return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows: break
            try:
                obj = json.loads(line.strip())
                for d in obj.get("detections", []) or []:
                    rows.append({
                        "t": d.get("t"),
                        "label": d.get("label"),
                        "conf": d.get("conf"),
                    })
            except Exception:
                continue
    return rows

@bp.get("")
def vision():
    return render_template("nav_page/vision.html")

# -------- 실행 --------
@bp.route("/run", methods=["POST"])
def run():
    # 파일 체크
    if "video_in" not in request.files:
        return jsonify({"ok": False, "error": "NO_FILE"}), 400
    f = request.files["video_in"]
    if not f or not _is_allowed(f.filename):
        return jsonify({"ok": False, "error": "INVALID_FILE"}), 400

    # 출력 디렉토리 준비
    job_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir_rel = os.path.join("vision", job_id)
    out_dir_abs = _static_path(out_dir_rel)
    _ensure_dir(out_dir_abs)

    # 입력 저장
    in_name = secure_filename(f.filename)
    in_path = os.path.join(out_dir_abs, in_name)
    f.save(in_path)

    # 출력/로그 경로
    out_video_name = "output.mp4"
    out_video_path = os.path.join(out_dir_abs, out_video_name)
    det_jsonl_path = os.path.join(out_dir_abs, "detections.jsonl")
    summary_path = os.path.join(out_dir_abs, "summary.json")

    # === 경로 보정 (repo 루트 기준 절대경로) ===
    # 현재 파일: wildfire/wildfire/views/vision_views.py
    # repo 루트: wildfire/
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ▲▲▲ 3단계
    detector_abs = os.path.join("detectors", "fire_smoke_detect.py")
    model_abs    = os.path.join(repo_root, "wildfire", "ML", "weights", "best.pt")
    # =======================================

    logs_all: List[str] = []
    processed_url = None
    best_detections: List[Dict[str, Any]] = []
    best_conf_used = None

    # conf 후보 순차 시도
    for conf in CONF_CANDIDATES:
        # 실행 전, 이전 결과물 삭제(덮어쓰기 안전)
        for p in (out_video_path, det_jsonl_path, summary_path):
            try:
                if os.path.isfile(p): os.remove(p)
            except Exception:
                pass

        cmd = [
            os.environ.get("PYTHON_BIN", os.sys.executable), "-u", detector_abs,
            "--model", model_abs,
            "--video-in", in_path,
            "--video-out", out_video_path,
            "--conf", f"{conf}",
            "--iou", f"{IOU_FIXED}",
            "--targets", TARGETS,
            "--save-jsonl", det_jsonl_path,
            "--summary", summary_path,
            "--imgsz", str(IMGSZ_DEFAULT),
            "--vid-stride", str(VID_STRIDE_DEFAULT),
            "--max-det", str(MAX_DET_DEFAULT),
        ]
        if AGNOSTIC_NMS_DEFAULT:
            cmd.append("--agnostic-nms")
        if DEVICE_FIXED:  # 비워두면 fire_smoke_detect.py가 자동 선택(cuda→mps→cpu)
            cmd.extend(["--device", DEVICE_FIXED])

        # 디버그: 실제 커맨드 기록
        logs_all.append("[CMD] " + " ".join(cmd))

        # CWD는 굳이 옮길 필요 없음(절대경로 사용)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        logs_all.extend((proc.stdout or "").splitlines())

        processed_ok = os.path.isfile(out_video_path)
        if processed_ok:
            processed_url = _static_url(os.path.join(out_dir_rel, out_video_name))

        dets_now = _read_detections_head(det_jsonl_path, max_rows=100)

        # 결과 반영
        best_detections = dets_now
        best_conf_used = conf

        # 탐지가 생기면 그 시점에서 종료
        if dets_now:
            break

    # 이벤트 요약(있으면)
    events = []
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
                events = s.get("events", []) or []
        except Exception:
            pass

    # 성공 판정: 주석 비디오가 생성되었으면 성공(탐지 유무와 무관)
    ok_flag = processed_url is not None

    return jsonify({
        "ok": ok_flag,
        "processed_url": processed_url,
        "detections": best_detections,
        "events": events,
        "used_conf": best_conf_used,
        "used_iou": IOU_FIXED,
        "job_id": job_id,
        "log_tail": logs_all[-80:],  # 디버그 tail
    })

@bp.route("/status", methods=["GET"])
def status():
    return jsonify({"ok": True})