from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from embeddings import EmbeddingService
from vectorstore import QdrantStore
from reranker import Reranker
from config import qdrant_config

class Retriever:
	def __init__(self, store: QdrantStore | None = None, embedder: EmbeddingService | None = None, reranker: Reranker | None = None):
		self.store = store or QdrantStore()
		self.embedder = embedder or EmbeddingService()
		rself = reranker or Reranker()
		self.reranker = rself

	def search(self, query: str, top_k: int = 20, mmr_k: int = 8, filter_: Optional[Any] = None, collection: Optional[str] = None) -> List[Dict[str, Any]]:
		collection_name = collection or qdrant_config.collection
		q_vec = self.embedder.embed_text(query)
		initial = self.store.query(collection_name, q_vec, top_k=top_k, filter_=filter_)
		if not initial:
			return []
		# MMR selection on embedding vectors
		vectors = np.array([q_vec] + [r["payload"].get("vector", q_vec) for r in initial], dtype=float)  # fallback if vector not stored
		# If vectors for docs are not stored, skip MMR
		selected = initial
		if vectors.shape[0] == len(initial) + 1 and vectors.shape[1] == len(q_vec):
			selected = self._mmr(vectors[0], vectors[1:], initial, k=min(mmr_k, len(initial)))
		return self.reranker.rerank(query, selected)

	@staticmethod
	def _cosine(a: np.ndarray, b: np.ndarray) -> float:
		na = np.linalg.norm(a)
		nb = np.linalg.norm(b)
		if na == 0 or nb == 0:
			return 0.0
		return float(np.dot(a, b) / (na * nb))

	def _mmr(self, q: np.ndarray, docs: np.ndarray, items: List[Dict[str, Any]], k: int, lambda_: float = 0.5) -> List[Dict[str, Any]]:
		selected_idx: List[int] = []
		candidates = list(range(len(docs)))
		while len(selected_idx) < k and candidates:
			scores = []
			for i in candidates:
				relevance = self._cosine(q, docs[i])
				diversity = max([self._cosine(docs[i], docs[j]) for j in selected_idx], default=0.0)
				score = lambda_ * relevance - (1 - lambda_) * diversity
				scores.append((score, i))
			scores.sort(reverse=True)
			sel = scores[0][1]
			selected_idx.append(sel)
			candidates.remove(sel)
		return [items[i] for i in selected_idx]
