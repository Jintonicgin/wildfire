# wildfire/rag_store.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# --------- 데이터 타입 ----------
@dataclass
class DocChunk:
    id: Optional[str]
    text: str
    meta: Dict


# --------- 임베딩 래퍼 ----------
class Embeddings:
    """
    SentenceTransformer 임베딩을 **항상 CPU**에서 수행하도록 고정.
    (Mac/MPS 및 meta 텐서 이슈 회피)
    """
    def __init__(self):
        model_id = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        # ✅ CPU 고정
        self.model = SentenceTransformer(model_id, device="cpu")
        self.normalize = True
        self.batch_size = int(os.getenv("EMBED_BATCH", "64"))

    def encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            device="cpu",  # ✅ 호출 시에도 CPU 고정
        )
        return vecs.tolist()


# --------- VectorStore 추상 ----------
class VectorStore:
    def ensure_collection(self, name: str, dim: int): ...
    def upsert(self, collection: str, chunks: List[DocChunk], vectors: List[List[float]]): ...
    def query(self, collection: str, query_vec: List[float], k: int,
              score_threshold: float = 0.0) -> List[Dict]: ...
    def delete_collection(self, name: str): ...
    def count(self, name: str) -> int: ...


# --------- Qdrant 구현 ----------
class QdrantStore(VectorStore):
    def __init__(self):
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        timeout = float(os.getenv("QDRANT_TIMEOUT", "20"))
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.batch_size = int(os.getenv("QDRANT_BATCH", "256"))

    def _get_existing_dim(self, info) -> Optional[int]:
        """Qdrant 버전에 따라 다른 위치에 있는 벡터 차원(size) 추출."""
        # 최신: info.config.params.vectors.size
        cfg = getattr(info, "config", None)
        if cfg and getattr(cfg, "params", None):
            vectors = getattr(cfg.params, "vectors", None)
            if vectors and hasattr(vectors, "size"):
                return vectors.size
        # 구버전 호환
        if hasattr(info, "vectors_config") and hasattr(info.vectors_config, "size"):
            return info.vectors_config.size
        return None

    def ensure_collection(self, name: str, dim: int):
        """
        - 컬렉션이 있으면 차원 확인 (불일치 시 명시적 에러 발생)
        - 없으면 COSINE 거리로 새로 생성
        - 텍스트 검색을 위해 text 필드에 payload 인덱스 보장
        """
        try:
            info = self.client.get_collection(name)
            exist_dim = self._get_existing_dim(info)
            if exist_dim is not None and exist_dim != dim:
                raise ValueError(
                    f"[Qdrant] Collection '{name}' dimension mismatch. "
                    f"exists={exist_dim}, need={dim}. "
                    f"Drop the collection or re-create with the correct embedding model."
                )
            # payload 인덱스 생성(이미 있으면 무시)
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="text",
                    field_schema=qmodels.PayloadSchemaType.TEXT,
                )
            except Exception:
                pass
            return
        except Exception:
            # 존재하지 않거나 get 실패 → 새로 생성
            self.client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE
                ),
            )
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="text",
                    field_schema=qmodels.PayloadSchemaType.TEXT,
                )
            except Exception:
                pass

    def upsert(self, collection: str, chunks: List[DocChunk], vectors: List[List[float]]):
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")

        pts: List[qmodels.PointStruct] = []
        for c, v in zip(chunks, vectors):
            pid = c.id or str(uuid4())
            payload = {"text": c.text}
            if c.meta:
                payload.update(c.meta)
            pts.append(qmodels.PointStruct(id=pid, vector=v, payload=payload))

        # 배치 업서트 (간단 재시도 1회)
        for i in range(0, len(pts), self.batch_size):
            batch = pts[i:i + self.batch_size]
            try:
                self.client.upsert(collection_name=collection, points=batch, wait=True)
            except Exception:
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
                "score": float(r.score),  # 벡터 유사도 점수
            })
        return out

    def delete_collection(self, name: str):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    def count(self, name: str) -> int:
        try:
            return int(self.client.count(name, exact=True).count)
        except Exception:
            return 0


def get_store() -> VectorStore:
    return QdrantStore()