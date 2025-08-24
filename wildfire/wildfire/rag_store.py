# rag_store.py
import os
from typing import List, Dict, Optional, Iterable
from dataclasses import dataclass
from uuid import uuid4

# Embeddings
from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# =========================
# Datatypes
# =========================
@dataclass
class DocChunk:
    id: Optional[str]
    text: str
    meta: Dict


# =========================
# Embedding wrapper
# =========================
class Embeddings:
    """
    SentenceTransformer 래퍼.
    - 모델은 ENV EMBED_MODEL 로 선택 (기본: multilingual-e5-base)
    - batch, device 자동 설정
    - normalize 옵션 기본 True
    """
    def __init__(self):
        model_id = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
        device = "cuda" if os.getenv("EMBED_DEVICE", "").lower() in {"cuda", "gpu"} else None
        # device=None 이면 sentence-transformers 가 자동 선택(cpu/cuda)
        self.model = SentenceTransformer(model_id, device=device)
        self.normalize = (os.getenv("EMBED_NORMALIZE", "1") not in {"0", "false", "False"})
        self.batch_size = int(os.getenv("EMBED_BATCH", "64"))

        # E5/BGE 스타일 모델 최적 prefix 사용 여부
        self.use_instruction = os.getenv("EMBED_INSTRUCTION", "1") not in {"0", "false", "False"}
        self.query_prefix = os.getenv("EMBED_QUERY_PREFIX", "query: ")
        self.doc_prefix = os.getenv("EMBED_DOC_PREFIX", "passage: ")

    def _maybe_prefix(self, texts: Iterable[str], kind: str) -> List[str]:
        if not self.use_instruction:
            return list(texts)
        prefix = self.query_prefix if kind == "query" else self.doc_prefix
        return [f"{prefix}{t}" for t in texts]

    def encode_docs(self, texts: List[str]) -> List[List[float]]:
        """문서/청크 임베딩"""
        texts = self._maybe_prefix(texts, kind="doc")
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def encode_query(self, text: str) -> List[float]:
        """쿼리 임베딩"""
        txt = self._maybe_prefix([text], kind="query")
        vec = self.model.encode(
            txt,
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0].tolist()

    @property
    def dim(self) -> int:
        # SentenceTransformer 는 첫 forward 전에도 get_sentence_embedding_dimension 제공
        return self.model.get_sentence_embedding_dimension()


# =========================
# Abstract VectorStore
# =========================
class VectorStore:
    def ensure_collection(self, name: str, dim: int): ...
    def upsert(self, collection: str, chunks: List[DocChunk], vectors: List[List[float]]): ...
    def query(self, collection: str, query_vec: List[float], k: int,
              score_threshold: float = 0.0) -> List[Dict]: ...
    def delete_collection(self, name: str): ...
    def count(self, name: str) -> int: ...


# =========================
# Qdrant implementation
# =========================
class QdrantStore(VectorStore):
    def __init__(self):
        """
        ENV
        - QDRANT_URL (예: http://localhost:6333)  또는
        - QDRANT_HOST, QDRANT_PORT
        - QDRANT_API_KEY (옵션)
        """
        url = os.getenv("QDRANT_URL")
        host = os.getenv("QDRANT_HOST")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY") or None
        timeout = float(os.getenv("QDRANT_TIMEOUT", "20"))

        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        else:
            self.client = QdrantClient(host=host or "localhost", port=port, api_key=api_key, timeout=timeout)

        # upsert batch size
        self.batch_size = int(os.getenv("QDRANT_BATCH", "256"))

    def ensure_collection(self, name: str, dim: int):
        """
        컬렉션 없으면 생성(있으면 유지). re/create 가 아닌 create_if_not_exists 패턴.
        """
        try:
            self.client.get_collection(name)
            return
        except Exception:
            pass

        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(
                default_segment_number=2
            ),
            # HNSW 파라미터는 기본값으로도 충분. 필요시 ENV로 노출.
        )

    def upsert(self, collection: str, chunks: List[DocChunk], vectors: List[List[float]]):
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) and vectors({len(vectors)}) length mismatch")

        # None id -> uuid 부여
        pts: List[qmodels.PointStruct] = []
        for c, v in zip(chunks, vectors):
            pid = c.id or str(uuid4())
            payload = {"text": c.text}
            if c.meta:
                payload.update(c.meta)
            pts.append(qmodels.PointStruct(id=pid, vector=v, payload=payload))

        # 배치 업서트
        for i in range(0, len(pts), self.batch_size):
            batch = pts[i:i + self.batch_size]
            self.client.upsert(collection_name=collection, points=batch, wait=True)

    def query(self, collection: str, query_vec: List[float], k: int,
              score_threshold: float = 0.0) -> List[Dict]:
        res = self.client.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=int(k),
            score_threshold=(score_threshold or None),
            with_payload=True,
        )
        out: List[Dict] = []
        for r in res:
            payload = r.payload or {}
            out.append({
                "id": str(r.id),
                "text": payload.get("text", ""),
                "meta": {k: v for k, v in payload.items() if k != "text"},
                "score": float(r.score),
            })
        return out

    def delete_collection(self, name: str):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    def count(self, name: str) -> int:
        try:
            info = self.client.get_collection(name)
            # points_count 는 info 에 바로 없을 수 있어, 데이터 스냅샷 사용
            snap = self.client.scroll(name, limit=1)
            # scroll 은 total 을 직접 안주므로 대략 확인용으로 다시 조회
            # 정확 카운트는 별도 API가 없어 간단히 estimate 로 대체
            # 필요시 payload filter 없이 large limit 로 쿼리 금물. 운영에선 별도 관리 권장.
            return int(info.vectors_count) if hasattr(info, "vectors_count") else len(snap[0])
        except Exception:
            return 0


# =========================
# Factory
# =========================
def get_store() -> VectorStore:
    kind = (os.getenv("VECTOR_DB") or "qdrant").lower()
    if kind == "qdrant":
        return QdrantStore()
    # elif kind == "pinecone": return PineconeStore(...)
    # elif kind == "weaviate": return WeaviateStore(...)
    return QdrantStore()


def get_embeddings() -> Embeddings:
    return Embeddings()