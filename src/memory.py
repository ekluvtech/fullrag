from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

from embeddings import EmbeddingService
from vectorstore import QdrantStore
from config import qdrant_config

@dataclass
class ChatMessage:
	role: str
	content: str
	timestamp: float = field(default_factory=lambda: time.time())

class ShortTermMemory:
	def __init__(self, max_messages: int = 20):
		self.max_messages = max_messages
		self.messages: List[ChatMessage] = []

	def add(self, role: str, content: str) -> None:
		self.messages.append(ChatMessage(role=role, content=content))
		if len(self.messages) > self.max_messages:
			self.messages = self.messages[-self.max_messages:]

	def get(self) -> List[Dict[str, Any]]:
		return [m.__dict__ for m in self.messages]

class LongTermMemory:
	def __init__(self, store: QdrantStore | None = None, embedder: EmbeddingService | None = None):
		self.store = store or QdrantStore()
		self.embedder = embedder or EmbeddingService()
		self.store.ensure_collection(qdrant_config.memory_collection, vector_size=self.embedder.model.get_sentence_embedding_dimension())

	def add(self, session_id: str, role: str, content: str) -> None:
		text = f"[{role}] {content}"
		vec = self.embedder.embed_text(text)
		payload = {"session_id": session_id, "role": role, "text": content}
		self.store.upsert(qdrant_config.memory_collection, [vec], [payload])

	def recall(self, session_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		vec = self.embedder.embed_text(query)
		filter_ = self.store.build_filter("session_id", session_id)
		return self.store.query(qdrant_config.memory_collection, vec, top_k=top_k, filter_=filter_)
