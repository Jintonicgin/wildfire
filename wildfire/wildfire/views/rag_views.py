from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import io
import uuid
from typing import List, Dict, Tuple
from wildfire.auth_utils import admin_required
from wildfire.rag_store import get_store, Embeddings, QdrantStore

bp = Blueprint("rag", __name__)

# ===== 환경설정 =====
RAG_COLLECTION   = os.getenv("RAG_COLLECTION", "wildfire_corpus")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "5"))
RAG_THRESHOLD    = float(os.getenv("RAG_SCORE_THRESHOLD", "0.25"))
RAG_CHUNK_SIZE   = int(os.getenv("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RAG_CORPUS_DIR   = os.getenv("RAG_CORPUS_DIR", "wildfire/dataset/corpus")

# 전역 인스턴스 (간단 캐시)
_store: QdrantStore = get_store()
_emb  = Embeddings()

# ===== 유틸: 텍스트 추출 =====
def _read_txt_or_md(fp: io.BytesIO, encoding="utf-8") -> str:
    return fp.read().decode(encoding, errors="ignore")

def _read_pdf(fp: io.BytesIO) -> str:
    try:
        import PyPDF2  # optional
    except Exception:
        return ""
    fp.seek(0)
    reader = PyPDF2.PdfReader(fp)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)

def _extract_text(filename: str, filebytes: bytes) -> str:
    name = (filename or "").lower()
    b = io.BytesIO(filebytes)
    if name.endswith(".txt") or name.endswith(".md"):
        return _read_txt_or_md(b)
    if name.endswith(".pdf"):
        return _read_pdf(b)
    return ""  # 미지원 확장자

# ===== 유틸: 청킹 =====
def _chunk(s: str, size: int, overlap: int) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    start = 0
    n = len(s)
    step = max(1, size - max(0, overlap))
    while start < n:
        out.append(s[start:start+size])
        start += step
    return out

# ===== 유틸: 업서트 =====
def _ensure_collection(dim: int):
    _store.ensure_collection(RAG_COLLECTION, dim)

def _upsert_texts(texts: List[str], meta_base: Dict):
    if not texts:
        return 0
    vecs = _emb.encode(texts)
    _ensure_collection(dim=len(vecs[0]))
    from wildfire.rag_store import DocChunk  # dataclass
    chunks = [
        DocChunk(id=str(uuid.uuid4()), text=t, meta=meta_base)
        for t in texts
    ]
    _store.upsert(RAG_COLLECTION, chunks, vecs)
    return len(chunks)

# ===== 페이지 =====
@bp.get("/rag")
@admin_required
def rag():
    return render_template("nav_page/rag.html")

# ===== API: 통계 =====
@bp.get("/api/rag/stats")
@admin_required
def rag_stats():
    try:
        # Qdrant 통계 (컬렉션이 없으면 예외 → 0 처리)
        try:
            info = _store.client.get_collection(RAG_COLLECTION)
            # count
            cnt = _store.client.count(RAG_COLLECTION, exact=True).count
        except Exception:
            info = None
            cnt = 0
        return jsonify({
            "ok": True,
            "stats": {
                "collection": RAG_COLLECTION,
                "points": cnt,
                "qdrant_info": (info.dict() if info else None)
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ===== API: 디렉토리 재인덱싱 =====
@bp.post("/api/rag/reindex")
@admin_required
def rag_reindex():
    """
    RAG_CORPUS_DIR 안의 .txt/.md/.pdf를 스캔 → 전체 재생성(간단히 recreate)
    """
    try:
        # recreate 대신: 내부 ensure_collection은 recreate가 아니라 새로 만들기만 함.
        # "완전 초기화"를 원하면 Qdrant의 recreate_collection을 직접 호출:
        from qdrant_client.http import models as qmodels
        # 임시로 임베딩 차원 파악
        probe_vec = _emb.encode(["probe"])[0]
        dim = len(probe_vec)
        # 완전 재생성
        _store.client.recreate_collection(
            collection_name=RAG_COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

        total = 0
        for root, _, files in os.walk(RAG_CORPUS_DIR):
            for fn in files:
                if not (fn.lower().endswith(".txt") or fn.lower().endswith(".md") or fn.lower().endswith(".pdf")):
                    continue
                path = os.path.join(root, fn)
                with open(path, "rb") as f:
                    text = _extract_text(fn, f.read())
                chunks = _chunk(text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLP)
                total += _upsert_texts(chunks, meta_base={"source": "corpus", "filename": fn})
        return jsonify({"ok": True, "indexed": total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ===== API: 파일 업로드 인덱싱 =====
@bp.post("/api/rag/index-files")
@admin_required
def rag_index_files():
    """
    관리자 업로드(.txt/.md/.pdf)로 증분 인덱싱
    """
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"ok": False, "error": "NO_FILES"}), 400

        added_total = 0
        for f in files:
            filename = secure_filename(f.filename or "uploaded")
            data = f.read()
            text = _extract_text(filename, data)
            if not text:
                continue
            chunks = _chunk(text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLP)
            added_total += _upsert_texts(chunks, meta_base={"source": "upload", "filename": filename})

        return jsonify({"ok": True, "added": added_total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ===== API: 검색 =====
@bp.get("/api/rag/search")
@admin_required
def rag_search():
    q = (request.args.get("q") or "").strip()
    top_k = int(request.args.get("k") or RAG_TOP_K)
    thr   = float(request.args.get("thr") or RAG_THRESHOLD)
    if not q:
        return jsonify({"ok": False, "error": "EMPTY_QUERY"}), 400
    try:
        qv = _emb.encode([q])[0]
        hits = _store.query(RAG_COLLECTION, qv, k=top_k, score_threshold=thr)
        # hits: [{"text":..., "meta":..., "score":...}]
        return jsonify({"ok": True, "results": hits})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500