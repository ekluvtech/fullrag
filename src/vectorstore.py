from __future__ import annotations
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from uuid import uuid4
from config import qdrant_config, embedding_config

class QdrantStore:
	def __init__(self, url: str | None = None, api_key: Optional[str] = None):
		self.client = QdrantClient(url=url or qdrant_config.url, api_key=api_key or qdrant_config.api_key)

	def ensure_collection(self, name: str, vector_size: int, distance: Distance = Distance.COSINE) -> None:
		collections = [c.name for c in self.client.get_collections().collections]
		if name not in collections:
			self.client.create_collection(collection_name=name, vectors_config=VectorParams(size=vector_size, distance=distance))

	def upsert(self, collection: str, embeddings: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
		points = []
		for vec, payload in zip(embeddings, payloads):
			points.append(PointStruct(id=str(uuid4()), vector=vec, payload=payload))
		self.client.upsert(collection_name=collection, points=points)

	def query(self, collection: str, vector: List[float], top_k: int = 20, filter_: Optional[Filter] = None) -> List[Dict[str, Any]]:
		search_result = self.client.search(collection_name=collection, query_vector=vector, limit=top_k, query_filter=filter_)
		results = []
		for r in search_result:
			item = {
				"id": r.id,
				"score": r.score,
				"payload": r.payload,
			}
			results.append(item)
		return results

	@staticmethod
	def build_filter(field: str, value: Any) -> Filter:
		return Filter(must=[FieldCondition(key=field, match=MatchValue(value=value))])


def init_default_collections(store: QdrantStore) -> None:
	store.ensure_collection(qdrant_config.collection, embedding_config.dimension)
	store.ensure_collection(qdrant_config.memory_collection, embedding_config.dimension)
