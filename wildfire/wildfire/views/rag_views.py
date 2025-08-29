# wildfire/views/rag_views.py
from flask import Blueprint, request, jsonify, render_template, Response, stream_with_context
import os
import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

bp = Blueprint("rag", __name__)

# ====== 환경설정 ======
# 클라우드 FastAPI(또는 Flask) 백엔드 베이스 URL
CLOUD_GAI_BASE       = os.getenv("GAI_BACKEND_URL", "https://lijpcw8himz39f-8080.proxy.runpod.net").rstrip("/")

# (선택) 클라우드 인증 전달 방법 – 필요 시만 세팅
CLOUD_BEARER_TOKEN   = os.getenv("CLOUD_BEARER_TOKEN", "").strip()
CLOUD_SESSION_COOKIE = os.getenv("CLOUD_SESSION_COOKIE", "").strip()

# 로컬 코퍼스 폴더 (업로드 파일은 여기 저장) — 현재 화면용만, 인덱싱은 클라우드가 수행
RAG_CORPUS_DIR       = os.getenv("RAG_CORPUS_DIR", "wildfire/corpus")

# 검색 파라미터 (프런트 기본값과 일치)
RAG_TOP_K            = int(os.getenv("RAG_TOP_K", "5"))
RAG_THRESHOLD        = float(os.getenv("RAG_SCORE_THRESHOLD", "0.25"))
RAG_SNIPPET_CHARS    = int(os.getenv("RAG_SNIPPET_CHARS", "500"))

# ====== 공통 유틸 ======
def _cloud_headers(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if CLOUD_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {CLOUD_BEARER_TOKEN}"
    if CLOUD_SESSION_COOKIE:
        headers["Cookie"] = CLOUD_SESSION_COOKIE
    if extra:
        headers.update(extra)
    return headers

def _cloud_get(path: str, **kwargs):
    """
    GET을 호출하고 (json, status_code)를 반환.
    상태코드가 200이 아니어도 예외를 던지지 않음.
    """
    url = f"{CLOUD_GAI_BASE}{path}"
    headers = _cloud_headers(kwargs.pop("headers", None))
    r = requests.get(url, headers=headers, timeout=kwargs.pop("timeout", 120), **kwargs)
    try:
        j = r.json()
    except Exception:
        j = {"ok": False, "error": r.text}
    return j, r.status_code

def _cloud_post(path: str, **kwargs):
    """
    POST를 호출하고 (json, status_code)를 반환.
    상태코드가 200이 아니어도 예외를 던지지 않음.
    """
    url = f"{CLOUD_GAI_BASE}{path}"
    headers = _cloud_headers(kwargs.pop("headers", None))
    r = requests.post(url, headers=headers, timeout=kwargs.pop("timeout", 300), **kwargs)
    try:
        j = r.json()
    except Exception:
        j = {"ok": False, "error": r.text}
    return j, r.status_code

# ====== 스트리밍 POST 프록시 ======
def _cloud_post_stream(path: str, json_body: dict, chunk_size: int = 1024):
    """
    클라우드 스트리밍 응답을 그대로 중계하는 제너레이터.
    """
    url = f"{CLOUD_GAI_BASE}{path}"
    headers = _cloud_headers({"Content-Type": "application/json"})
    try:
        r = requests.post(url, headers=headers, json=json_body, stream=True, timeout=300)
    except requests.RequestException as e:
        # 연결/타임아웃 에러를 즉시 JSON으로 반환
        yield (jsonify({"ok": False, "error": f"STREAM_CONNECT_FAILED: {e}"}).get_data(as_text=True)).encode("utf-8")
        return

    # 상태 코드가 200이 아니면 본문을 한 번에 읽어 에러로 리턴
    if not r.ok or r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"ok": False, "error": r.text}
        yield (jsonify(err).get_data(as_text=True)).encode("utf-8")
        return

    # OK: 청크를 그대로 전달
    for chunk in r.iter_content(chunk_size=chunk_size):
        if chunk:
            yield chunk

def _build_context_from_hits(hits: List[Dict[str, Any]], max_chars_each: int) -> Dict[str, Any]:
    """
    hits: [{"text": "...", "score": ..., "rerank_score": ..., "meta": {"source":..., "filename":...}}]
    → {"context": "...", "sources":[...]}
    (현재는 클라우드가 context/sources를 반환하므로 예비용)
    """
    ctx_lines: List[str] = []
    sources: List[Dict[str, Any]] = []
    for i, h in enumerate(hits, start=1):
        t = (h.get("text") or "")
        if len(t) > max_chars_each:
            t = t[:max_chars_each] + "…"
        meta = h.get("meta", {}) or {}
        src = meta.get("source") or meta.get("filename") or h.get("source") or "doc"
        sid = f"S{i}"
        ctx_lines.append(f"[{sid}] ({src})\n{t}\n")
        sources.append({
            "id": sid,
            "source": src,
            "score": h.get("score"),
            "rerank_score": h.get("rerank_score"),
            "text": t
        })
    return {"context": "\n".join(ctx_lines).strip(), "sources": sources}

# ====== 페이지 ======
@bp.get("/rag")
def rag():
    return render_template("nav_page/rag.html")

# ====== 검색: 클라우드에 위임 (재랭킹 포함) ======
@bp.get("/api/rag/search")
def rag_search():
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k") or RAG_TOP_K)
    thr = float(request.args.get("thr") or RAG_THRESHOLD)
    if not q:
        return jsonify({"ok": False, "error": "EMPTY_QUERY"}), 400
    try:
        j, status = _cloud_post("/api/rag/search", json={"query": q, "k": k, "thr": thr}, timeout=90)
        # 클라우드가 200 이외 상태(예: 422)를 내리더라도 프런트에서 그대로 처리할 수 있게 그대로 전달
        return jsonify(j), status
    except Exception as e:
        logger.exception("rag_search failed")
        return jsonify({"ok": False, "error": str(e)}), 500

# ====== 질문: 클라우드에서 검색+생성+출처 전부 처리 ======
@bp.post("/api/rag/ask")
def rag_ask():
    try:
        data = request.get_json(silent=True) or {}
        q = (data.get("query") or "").strip()
        k = int(data.get("k") or RAG_TOP_K)
        thr = float(data.get("thr") or RAG_THRESHOLD)
        if not q:
            return jsonify({"ok": False, "error": "EMPTY_QUERY"}), 400

        # 클라우드에서 검색+생성 모두 수행
        j, status = _cloud_post("/api/rag/ask", json={"query": q, "k": k, "thr": thr}, timeout=180)

        # 프런트 rag.js의 postJSON은 non-200을 throw하므로
        # 사용자 오류형(422 등)은 200으로 내려주어 UI가 data.error로 안내문을 띄우도록 함
        if status == 422:
            return jsonify(j), 200

        # 그 외 상태는 그대로 전달 (예: 500 등 서버 오류)
        return jsonify(j), status

    except Exception as e:
        logger.exception("rag_ask failed")
        return jsonify({"ok": False, "error": str(e)}), 500

# ====== 질문 스트리밍: 클라우드에서 검색+생성 스트림 중계 ======
@bp.post("/api/rag/ask_stream")
def rag_ask_stream():
    try:
        data = request.get_json(silent=True) or {}
        q   = (data.get("query") or "").strip()
        k   = int(data.get("k") or RAG_TOP_K)
        thr = float(data.get("thr") or RAG_THRESHOLD)
        if not q:
            return jsonify({"ok": False, "error": "EMPTY_QUERY"}), 400

        gen = _cloud_post_stream("/api/rag/ask_stream", {"query": q, "k": k, "thr": thr})
        # 스트리밍으로 바로 전달 (첫 바이트를 빨리 밀어 524 회피)
        return Response(stream_with_context(gen), mimetype="application/json; charset=utf-8")
    except Exception as e:
        logger.exception("rag_ask_stream failed")
        return jsonify({"ok": False, "error": str(e)}), 500