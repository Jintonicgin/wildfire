from __future__ import annotations

import io
import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Iterable, Optional

# ---- 텍스트 추출: pdfminer.six 우선 ----
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None

# ---- OCR fallback (이미지 PDF일 때) ----
try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:
    pytesseract = None
    convert_from_bytes = None

# ---- 임베딩 / 스토어 ----
from sentence_transformers import SentenceTransformer
from .rag_store import QdrantStore, DocChunk

# ---- 환경 ----
EMBED_MODEL     = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_COLLECTION  = os.getenv("RAG_COLLECTION", "wildfire_corpus")
CHUNK_SIZE      = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

# 번역은 기본 OFF (원문 그대로 인덱싱 권장)
TRANSLATE_TO_KO = (os.getenv("RAG_TRANSLATE_TO_KO", "false").lower() == "true")
# (원한다면 NLLB 붙여서 번역 on 할 수 있는데, 지금은 품질 위해 off 권장)

# ---------- 텍스트 추출 ----------
def extract_text_pdf_bytes(data: bytes) -> str:
    """pdfminer로 먼저 시도, 실패/빈문자면 OCR로 폴백"""
    # A) pdfminer (텍스트 레이어 존재 시 가장 깔끔)
    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(io.BytesIO(data))
            if txt and txt.strip():
                return txt
        except Exception:
            pass

    # B) OCR (영문 기준 – NASA/FWI 자료 등)
    if pytesseract and convert_from_bytes:
        try:
            images = convert_from_bytes(data, dpi=300)
            parts: List[str] = []
            for img in images:
                parts.append(pytesseract.image_to_string(img, lang="eng"))
            return "\n".join(parts)
        except Exception:
            pass

    return ""

# ---------- 클린업 ----------
_ws_re = re.compile(r"[ \t]+")
_hyphen_break_re = re.compile(r"-\n")                 # 하이픈 줄바꿈 이어붙이기
_dup_token_re = re.compile(r"(\b\w{2,}\b)(?:\s+\1){2,}")  # 같은 토큰이 3회 이상 반복되면 1회로 축소

def cleanup_text(s: str) -> str:
    s = s.replace("\ufeff", " ")
    s = _hyphen_break_re.sub("", s)      # "hyphen-\nwrap" -> "hyphenwrap"
    s = s.replace("\r", "\n")
    s = _ws_re.sub(" ", s)
    s = _dup_token_re.sub(r"\1", s)
    return s.strip()

def chunk_text(s: str, size: int, overlap: int) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[str] = []
    start, n = 0, len(s)
    step = max(1, size - max(0, overlap))
    while start < n:
        out.append(s[start:start + size])
        start += step
    return out

def is_usable_chunk(t: str) -> bool:
    """짧거나 기호·노이즈 위주 청크 걸러냄"""
    if len(t) < 80:
        return False
    letters = sum(c.isalpha() for c in t)
    ratio = letters / max(1, len(t))
    return ratio >= 0.35  # 알파비율 35% 미만이면 버림

# ---------- 배치 인입 ----------
@dataclass
class IngestStats:
    file_count: int
    chunk_count: int

class Ingestor:
    def __init__(self, embed_model: str = EMBED_MODEL, collection: str = RAG_COLLECTION):
        self.embedder = SentenceTransformer(embed_model)
        self.store = QdrantStore()
        self.collection = collection

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        return vecs.tolist() if hasattr(vecs, "tolist") else vecs

    def ingest_pdf_bytes(self, data: bytes, filename: str) -> int:
        raw = extract_text_pdf_bytes(data)
        if not raw.strip():
            return 0

        text = cleanup_text(raw)

        # (선택) 번역 – 품질 이슈 많아 기본 OFF 권장
        # if TRANSLATE_TO_KO: text = translate_to_ko(text)

        chunks = [c for c in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP) if is_usable_chunk(c)]
        if not chunks:
            return 0

        vecs = self._embed(chunks)
        self.store.ensure_collection(self.collection, dim=len(vecs[0]))
        points = [
            DocChunk(id=str(uuid.uuid4()), text=t, meta={"source": "pdf", "filename": filename})
            for t in chunks
        ]
        self.store.upsert(self.collection, points, vecs)
        return len(points)

    def ingest_paths(self, paths: Iterable[str]) -> IngestStats:
        fcount, ccount = 0, 0
        for p in paths:
            if not p.lower().endswith(".pdf"):
                continue
            with open(p, "rb") as f:
                added = self.ingest_pdf_bytes(f.read(), filename=os.path.basename(p))
            if added:
                fcount += 1
                ccount += added
        return IngestStats(file_count=fcount, chunk_count=ccount)

# CLI 사용 예: python -m wildfire.rag_ingest ./wildfire/corpus/foo.pdf ...
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m wildfire.rag_ingest <pdf1> <pdf2> ...")
        raise SystemExit(1)
    ing = Ingestor()
    stats = ing.ingest_paths(sys.argv[1:])
    print(f"[RAG Ingest] files={stats.file_count}, chunks={stats.chunk_count}")