# -*- coding: utf-8 -*-
import os, json, uuid, subprocess
from datetime import datetime
from typing import List, Dict, Any
from flask import Blueprint, request, jsonify, current_app, render_template, url_for
from werkzeug.utils import secure_filename

bp = Blueprint("vision", __name__, url_prefix="/vision")

# -------- 설정 --------
ALLOWED_EXTS = {"mp4", "mov", "m4v", "avi", "mkv", "webm"}

# 고정 모델(네가 학습한 가중치)
MODEL_FIXED = "models/yolov8s_fire_smoke_best.pt"
IOU_FIXED = 0.50
CONF_CANDIDATES = [0.25, 0.20, 0.15, 0.12]  # 위부터 순차 시도
TARGETS = "fire,smoke"

# -------- 유틸 --------
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

@bp.get("/vision")
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
    job_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
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

    # detector 스크립트(views/의 형제 디렉터리 detectors/ 안에 두었다면)
    mod_dir = os.path.dirname(os.path.abspath(__file__))               # .../views
    detector_path = os.path.join(os.path.dirname(mod_dir), "detectors", "fire_smoke_detect.py")

    # 모델 절대경로 보정
    model_abs = MODEL_FIXED
    if not os.path.isabs(model_abs):
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../wildfire
        model_abs = os.path.join(app_root, MODEL_FIXED) if MODEL_FIXED.startswith("wildfire/") else os.path.join(app_root, MODEL_FIXED)

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
            os.environ.get("PYTHON_BIN", os.sys.executable), "-u", detector_path,
            "--model", model_abs,
            "--video-in", in_path,
            "--video-out", out_video_path,
            "--conf", f"{conf}",
            "--iou", f"{IOU_FIXED}",
            "--targets", TARGETS,
            "--save-jsonl", det_jsonl_path,
            "--summary", summary_path,
        ]
        # 디버그: 실제 커맨드 기록
        logs_all.append("[CMD] " + " ".join(cmd))

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