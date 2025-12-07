from __future__ import annotations
from typing import List, Dict
from sentence_transformers import CrossEncoder
from config import reranker_config

class Reranker:
	def __init__(self, model_name: str | None = None):
		self.model_name = model_name or reranker_config.model_name
		self.model = CrossEncoder(self.model_name)

	def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
		pairs = [(query, c["payload"]["text"]) for c in candidates]
		scores = self.model.predict(pairs)
		for c, s in zip(candidates, scores):
			c["rerank_score"] = float(s)
		return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
