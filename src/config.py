import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class AppConfig(BaseModel):
	app_title: str = os.getenv("APP_TITLE", "Full RAG Chat")
	chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
	chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
	max_workers: int = int(os.getenv("MAX_WORKERS", "8"))

class QdrantConfig(BaseModel):
	url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
	api_key: str | None = os.getenv("QDRANT_API_KEY")
	collection: str = os.getenv("QDRANT_COLLECTION", "hc_data")
	memory_collection: str = os.getenv("QDRANT_MEMORY_COLLECTION", "chat_memory")

class EmbeddingConfig(BaseModel):
	mdel_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
	#model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/gte-large-en-v1.5")
	dimension: int = int(os.getenv("EMBEDDING_DIM", "384"))

class RerankerConfig(BaseModel):
	mdel_name: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

class LLMConfig(BaseModel):
	provider: str = os.getenv("LLM_PROVIDER", "ollama")
	ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
	model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
	temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))

class JudgeConfig(BaseModel):
	enabled: bool = os.getenv("JUDGE_ENABLED", "true").lower() == "true"
	model: str = os.getenv("JUDGE_MODEL", "llama3.2")
	threshold: float = float(os.getenv("JUDGE_THRESHOLD", "6.0"))

app_config = AppConfig()
qdrant_config = QdrantConfig()
embedding_config = EmbeddingConfig()
reranker_config = RerankerConfig()
llm_config = LLMConfig()
judge_config = JudgeConfig()
