from __future__ import annotations
from typing import List, Dict, Any, Iterable
from math import ceil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from embeddings import EmbeddingService
from vectorstore import QdrantStore
from config import qdrant_config, app_config

BATCH_SIZE = 512

class IngestionPipeline:
	def __init__(self, store: QdrantStore | None = None, embedder: EmbeddingService | None = None):
		self.store = store or QdrantStore()
		self.embedder = embedder or EmbeddingService()

	def _process_batch(self, collection_name: str, batch: List[Dict[str, Any]]) -> int:
		texts = [b["text"] for b in batch]
		payloads = [b["metadata"] | {"text": b["text"]} for b in batch]
		embeddings = self.embedder.embed_texts(texts)
		self.store.upsert(collection_name, embeddings, payloads)
		return len(batch)

	def ingest(self, chunks: List[Dict[str, Any]], collection: str | None = None) -> None:
		collection_name = collection or qdrant_config.collection
		self.store.ensure_collection(collection_name, vector_size=self.embedder.dimension)
		
		if not chunks:
			return
		n = len(chunks)
		n_batches = ceil(n / BATCH_SIZE)
		batches: List[List[Dict[str, Any]]] = []
		for i in range(n_batches):
			start = i * BATCH_SIZE
			end = min((i + 1) * BATCH_SIZE, n)
			batches.append(chunks[start:end])
		
		completed = 0
		with ThreadPoolExecutor(max_workers=app_config.max_workers) as executor:
			futures = [executor.submit(self._process_batch, collection_name, batch) for batch in batches]
			for f in tqdm(as_completed(futures), total=len(futures), desc="Upserting to Qdrant (parallel)"):
				completed += f.result()

	def ingest_stream(self, chunk_iter: Iterable[Dict[str, Any]], collection: str | None = None, max_in_flight: int | None = None) -> None:
		collection_name = collection or qdrant_config.collection
		self.store.ensure_collection(collection_name, vector_size=self.embedder.dimension)
		max_in_flight = max_in_flight or app_config.max_workers
		
		batch: List[Dict[str, Any]] = []
		in_flight = []
		total_completed = 0	
		with ThreadPoolExecutor(max_workers=app_config.max_workers) as executor:
			pbar = tqdm(desc="Streaming ingest", unit="chunks")
			for chunk in chunk_iter:
				batch.append(chunk)
				if len(batch) >= BATCH_SIZE:
					in_flight.append(executor.submit(self._process_batch, collection_name, batch))
					batch = []
					if len(in_flight) >= max_in_flight:
						done = next(as_completed(in_flight))
						total_completed += done.result()
						pbar.update(total_completed - pbar.n)
			# flush remaining batch
			if batch:
				in_flight.append(executor.submit(self._process_batch, collection_name, batch))
			# drain all
			for fut in as_completed(in_flight):
				total_completed += fut.result()
				pbar.update(total_completed - pbar.n)
			pbar.close()
