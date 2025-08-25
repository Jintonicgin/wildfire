# wildfire/views/rag_views.py
from flask import Blueprint, request, jsonify, render_template, g
import os
from typing import List

from wildfire.auth_utils import admin_required

bp = Blueprint("rag", __name__)

# ===== 환경설정 =====
RAG_COLLECTION   = os.getenv("RAG_COLLECTION", "wildfire_corpus")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "5"))
RAG_THRESHOLD    = float(os.getenv("RAG_SCORE_THRESHOLD", "0.25"))
RAG_CORPUS_DIR   = os.getenv("RAG_CORPUS_DIR", "wildfire/corpus")  # 서버 일괄 인덱싱 대상

# ===== Lazy creators (요청 스코프) =====
def get_store_lazy():
    """
    QdrantStore를 처음 접근할 때만 생성.
    """
    if not hasattr(g, "_rag_store"):
        from wildfire.rag_store import get_store  # 무거운 import 지연
        g._rag_store = get_store()
    return g._rag_store

def get_emb_lazy():
    """
    SentenceTransformer 래퍼(쿼리 임베딩용)를 처음 접근할 때만 생성.
    """
    if not hasattr(g, "_rag_emb"):
        from wildfire.rag_store import Embeddings
        g._rag_emb = Embeddings()
    return g._rag_emb

def get_ingestor_lazy():
    """
    업로드/재인덱싱용 Ingestor를 처음 접근할 때만 생성.
    (내부적으로 SentenceTransformer/NLLB 등을 초기화하므로 지연)
    """
    if not hasattr(g, "_rag_ingestor"):
        from wildfire.rag_ingest import Ingestor
        g._rag_ingestor = Ingestor()
    return g._rag_ingestor


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
        store = get_store_lazy()
        try:
            info = store.client.get_collection(RAG_COLLECTION)
            cnt = store.client.count(RAG_COLLECTION, exact=True).count
        except Exception:
            info, cnt = None, 0

        return jsonify({
            "ok": True,
            "stats": {
                "collection": RAG_COLLECTION,
                "points": cnt,
                "qdrant_info": (info.dict() if info else None),
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ===== API: 디렉토리 재인덱싱 =====
@bp.post("/api/rag/reindex")
@admin_required
def rag_reindex():
    """
    서버 디렉토리(RAG_CORPUS_DIR)의 PDF들을 다시 스캔해서 컬렉션을 완전히 재구축.
    """
    try:
        # 1) 컬렉션 완전 재생성 (차원 파악을 위해 probe 임베딩)
        emb = get_emb_lazy()
        probe_vec = emb.encode(["probe"])[0]
        dim = len(probe_vec)

        store = get_store_lazy()
        from qdrant_client.http import models as qmodels
        store.client.recreate_collection(
            collection_name=RAG_COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

        # 2) 디렉토리 내 PDF 수집
        paths: List[str] = []
        for root, _, files in os.walk(RAG_CORPUS_DIR):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    paths.append(os.path.join(root, fn))

        # 3) 일괄 인덱싱
        ing = get_ingestor_lazy()
        stats = ing.ingest_paths(paths)

        return jsonify({"ok": True, "indexed_files": stats.file_count, "indexed_chunks": stats.chunk_count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ===== API: 파일 업로드 인덱싱 =====
@bp.post("/api/rag/index-files")
@admin_required
def rag_index_files():
    """
    관리자 업로드(.pdf) 파일을 받아 corpus 디렉토리에 저장 후 즉시 인덱싱.
    """
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"ok": False, "error": "NO_FILES"}), 400

        # corpus 디렉토리 생성 (없다면)
        corpus_dir = os.path.join("wildfire", "wildfire", "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        ing = get_ingestor_lazy()
        added_total = 0
        saved_files = []

        for f in files:
            filename = f.filename or "uploaded.pdf"
            if not filename.lower().endswith(".pdf"):
                # 필요하면 .txt/.md 지원 추가 가능
                continue
            
            # 파일 데이터 읽기
            data = f.read()
            
            # 중복 파일명 처리 (timestamp 추가)
            import datetime
            base_name, ext = os.path.splitext(filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{base_name}_{timestamp}{ext}"
            
            # corpus 디렉토리에 파일 저장
            file_path = os.path.join(corpus_dir, safe_filename)
            with open(file_path, "wb") as corpus_file:
                corpus_file.write(data)
            
            # 벡터DB에 인덱싱
            added = ing.ingest_pdf_bytes(data, filename=safe_filename)
            added_total += added
            
            saved_files.append({
                "original_name": filename,
                "saved_name": safe_filename,
                "chunks": added
            })

        return jsonify({
            "ok": True, 
            "added": added_total,
            "files": saved_files,
            "corpus_path": corpus_dir
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ===== API: corpus 파일 목록 =====
@bp.get("/api/rag/corpus-files")
@admin_required
def rag_corpus_files():
    """
    corpus 디렉토리의 파일 목록 반환
    """
    try:
        corpus_dir = os.path.join("wildfire", "wildfire", "corpus")
        if not os.path.exists(corpus_dir):
            return jsonify({"ok": True, "files": []})
        
        files = []
        for filename in os.listdir(corpus_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(corpus_dir, filename)
                stat = os.stat(file_path)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        
        # 최신 파일 먼저 정렬
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({"ok": True, "files": files, "count": len(files)})
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
        emb = get_emb_lazy()
        store = get_store_lazy()

        qv = emb.encode([q])[0]
        hits = store.query(RAG_COLLECTION, qv, k=top_k, score_threshold=thr)
        # hits: [{"id","text","meta","score"}]
        return jsonify({"ok": True, "results": hits})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500