import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from uuid import uuid4

# --------- 데이터 타입 ----------
@dataclass
class DocChunk:
    id: Optional[str]
    text: str
    meta: Dict

# --------- 임베딩 래퍼 (필요시 사용) ----------
class Embeddings:
    def __init__(self):
        model_id = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_id)
        self.normalize = True
        self.batch_size = int(os.getenv("EMBED_BATCH", "64"))

    def encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=self.normalize, convert_to_numpy=True
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

    def ensure_collection(self, name: str, dim: int):
        try:
            self.client.get_collection(name)
            return
        except Exception:
            pass
        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, collection: str, chunks: List[DocChunk], vectors: List[List[float]]):
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        pts: List[qmodels.PointStruct] = []
        for c, v in zip(chunks, vectors):
            pid = c.id or str(uuid4())
            payload = {"text": c.text}
            if c.meta:
                payload.update(c.meta)
            pts.append(qmodels.PointStruct(id=pid, vector=v, payload=payload))
        for i in range(0, len(pts), self.batch_size):
            self.client.upsert(collection_name=collection, points=pts[i:i + self.batch_size], wait=True)

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
            return int(getattr(info, "vectors_count", 0))  # 근사치
        except Exception:
            return 0

def get_store() -> VectorStore:
    return QdrantStore()