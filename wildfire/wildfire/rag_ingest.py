# wildfire/rag_ingest.py
from __future__ import annotations

import io
import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Iterable

# ---- 텍스트 추출: pypdf 우선 ----
try:
    from pypdf import PdfReader  # type: ignore
    pypdf_available = True
except Exception:
    pypdf_available = False

# ---- 텍스트 추출: pdfminer.six 대안 ----
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

# ===================== 환경 =====================
EMBED_MODEL        = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RAG_COLLECTION     = os.getenv("RAG_COLLECTION", "wildfire_corpus")

# bge-m3 한글 문서 기준 권장(문자 단위)
CHUNK_SIZE         = int(os.getenv("RAG_CHUNK_SIZE", "1400"))
CHUNK_OVERLAP      = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# 접두어 정책 (E5/BGE 계열일 때 유용)
EMBED_USE_INSTR    = os.getenv("EMBED_INSTRUCTION", "0").lower() not in {"0", "false", "no"}
EMBED_DOC_PREFIX   = os.getenv("EMBED_DOC_PREFIX", "").strip()  # 예: "passage:"

# 번역은 기본 OFF (원문 그대로 인덱싱 권장)
TRANSLATE_TO_KO    = os.getenv("RAG_TRANSLATE_TO_KO", "false").lower() == "true"

# OCR 제어
OCR_ENABLE         = os.getenv("OCR_ENABLE", "false").lower() == "true"
OCR_LANG           = os.getenv("OCR_LANG", "kor+eng")
OCR_DPI            = int(os.getenv("OCR_DPI", "300"))
OCR_MAX_PAGES      = int(os.getenv("OCR_MAX_PAGES", "10"))  # 과도한 비용 방지

# 업서트 배치 크기
BATCH_SIZE         = int(os.getenv("RAG_INGEST_BATCH", "256"))

# 텍스트/마크다운 지원
_TEXT_EXTS = {".txt", ".md", ".markdown"}

# ===================== 텍스트 추출 =====================
def extract_text_pdf_bytes(data: bytes) -> str:
    """pypdf -> pdfminer -> (옵션) OCR 순으로 텍스트 추출"""
    # A) pypdf
    if pypdf_available:
        try:
            reader = PdfReader(io.BytesIO(data))
            parts: List[str] = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if t.strip():
                    parts.append(t)
            txt = "\n".join(parts)
            if txt.strip():
                print(f"[ingest] pypdf OK: {len(txt)} chars")
                return txt
        except Exception as e:
            print(f"[ingest] pypdf failed: {e}")

    # B) pdfminer
    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(io.BytesIO(data))
            if txt and txt.strip():
                print(f"[ingest] pdfminer OK: {len(txt)} chars")
                return txt
        except Exception as e:
            print(f"[ingest] pdfminer failed: {e}")

    # C) OCR (옵션)
    if OCR_ENABLE and pytesseract and convert_from_bytes:
        try:
            print("[ingest] OCR trying...")
            images = convert_from_bytes(data, dpi=OCR_DPI)
            if OCR_MAX_PAGES and len(images) > OCR_MAX_PAGES:
                images = images[:OCR_MAX_PAGES]
            parts: List[str] = []
            for i, img in enumerate(images):
                print(f"[ingest] OCR page {i+1}/{len(images)}")
                parts.append(pytesseract.image_to_string(img, lang=OCR_LANG))
            txt = "\n".join(parts)
            if txt and txt.strip():
                print(f"[ingest] OCR OK: {len(txt)} chars")
                return txt
        except Exception as e:
            print(f"[ingest] OCR failed: {e}")

    print("[ingest] ❌ text extraction failed")
    return ""

def extract_text_textlike_bytes(data: bytes) -> str:
    """txt/md 등의 바이트를 안전하게 디코드"""
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    # 최후 fallback
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ===================== 클린업 & 청킹 =====================
_ws_re = re.compile(r"[ \t]+")
_hyphen_break_re = re.compile(r"-\n")                   # 하이픈 줄바꿈 이어붙이기
_dup_token_re = re.compile(r"(\b\w{2,}\b)(?:\s+\1){2,}")  # 같은 토큰 3회 이상 반복 → 1회로 축소

def cleanup_text(s: str) -> str:
    s = s.replace("\ufeff", " ")
    s = _hyphen_break_re.sub("", s)   # "hyphen-\nwrap" -> "hyphenwrap"
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

# ===================== 통계 =====================
@dataclass
class IngestStats:
    file_count: int
    chunk_count: int

# ===================== 인게스터 =====================
class Ingestor:
    """
    - 임베딩은 **항상 CPU**에서 수행하여 MPS/meta 텐서 관련 에러 회피
    - bge-m3(1024차원) 등 대형 모델과 Qdrant 차원 불일치 방지:
      ensure_collection에서 차원 자동 확인/생성 (rag_store.QdrantStore)
    """
    def __init__(self, embed_model: str = EMBED_MODEL, collection: str = RAG_COLLECTION):
        # ✅ CPU 고정
        self.embedder = SentenceTransformer(embed_model, device="cpu")
        self.store = QdrantStore()
        self.collection = collection

    def _embed(self, texts: List[str]) -> List[List[float]]:
        X = texts
        # 문서 임베딩 접두어 (예: "passage: ...")
        if EMBED_USE_INSTR and EMBED_DOC_PREFIX:
            X = [f"{EMBED_DOC_PREFIX}{t}" for t in texts]

        vecs = self.embedder.encode(
            X,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.tolist() if hasattr(vecs, "tolist") else vecs

    # ---- 공개 API ----
    def ingest_pdf_bytes(self, data: bytes, filename: str) -> int:
        raw = extract_text_pdf_bytes(data)
        return self._ingest_text(raw, filename=filename, source="pdf")

    def ingest_text_bytes(self, data: bytes, filename: str) -> int:
        raw = extract_text_textlike_bytes(data)
        return self._ingest_text(raw, filename=filename, source="text")

    def ingest_paths(self, paths: Iterable[str]) -> IngestStats:
        fcount, ccount = 0, 0
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except Exception as e:
                print(f"[ingest] skip (read error): {p} - {e}")
                continue

            if ext == ".pdf":
                added = self.ingest_pdf_bytes(data, filename=os.path.basename(p))
            elif ext in _TEXT_EXTS:
                added = self.ingest_text_bytes(data, filename=os.path.basename(p))
            else:
                print(f"[ingest] skip (unsupported ext): {p}")
                continue

            if added:
                fcount += 1
                ccount += added

        return IngestStats(file_count=fcount, chunk_count=ccount)

    # ---- 내부 로직 ----
    def _ingest_text(self, raw: str, filename: str, source: str) -> int:
        if not (raw or "").strip():
            return 0

        text = cleanup_text(raw)

        # (선택) 번역 – 품질 이슈 많아 기본 OFF 권장
        # if TRANSLATE_TO_KO:
        #     text = translate_to_ko(text)

        chunks = [c for c in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP) if is_usable_chunk(c)]
        if not chunks:
            return 0

        vecs = self._embed(chunks)
        self.store.ensure_collection(self.collection, dim=len(vecs[0]))

        # 메타데이터 보강
        doc_id = str(uuid.uuid4())
        points: List[DocChunk] = []
        for idx, t in enumerate(chunks):
            points.append(
                DocChunk(
                    id=str(uuid.uuid4()),
                    text=t,
                    meta={
                        "source": source,
                        "filename": filename,
                        "doc_id": doc_id,
                        "chunk_index": idx,
                    },
                )
            )

        # 배치 업서트
        total = 0
        if BATCH_SIZE and BATCH_SIZE > 0:
            for i in range(0, len(points), BATCH_SIZE):
                batch_points = points[i:i + BATCH_SIZE]
                batch_vecs = vecs[i:i + BATCH_SIZE]
                self.store.upsert(self.collection, batch_points, batch_vecs)
                total += len(batch_points)
        else:
            self.store.upsert(self.collection, points, vecs)
            total = len(points)

        return total

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m wildfire.rag_ingest <file1> <file2> ...  (pdf/txt/md)")
        raise SystemExit(1)
    ing = Ingestor()
    stats = ing.ingest_paths(sys.argv[1:])
    print(f"[RAG Ingest] files={stats.file_count}, chunks={stats.chunk_count}")