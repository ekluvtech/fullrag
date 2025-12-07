from __future__ import annotations
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from config import embedding_config

class EmbeddingService:
	def __init__(self, model_name: str | None = None):
		self.model_name = model_name or embedding_config.mdel_name
		self.dimension = embedding_config.dimension
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = SentenceTransformer(self.model_name, device=self.device)

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		if not texts:
			return []
		embeddings = self.model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
		return embeddings.tolist()

	def embed_text(self, text: str) -> List[float]:
		return self.embed_texts([text])[0]



# from sentence_transformers import SentenceTransformer, InputExample, losses
# from torch.utils.data import DataLoader
# import torch

# # Load pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Sample training data (replace with your 1GB subset for fine-tuning)
# train_examples = [
#     InputExample(texts=['query1', 'relevant_doc1'], label=1.0),
#     # Add ~10k examples from your data
# ]

# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# train_loss = losses.CosineSimilarityLoss(model=model)

# # Fine-tune
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
# model.save('fine_tuned_embeddings')  # Save for later use


