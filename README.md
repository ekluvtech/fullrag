# Full RAG Implementation (Qdrant + Fine-tuned Embeddings + Reranking)

## Quickstart
1. Copy environment file:
```bash
cp env.example .env
```
2. Start Qdrant:
```bash
docker compose up -d qdrant
```
3. Install Ollama and pull a model (example: llama3.1:8b):
```bash
# macOS (brew): brew install ollama && ollama serve
# Or see https://ollama.com/download
ollama pull llama3.1:8b
```
4. Create venv and install deps:
```bash
 python3.10 -m venv fullrag && source fullrag/bin/activate
 pip install -r requirements.txt
```
5. Ingest your data directory (supports PDFs, DOCX, HTML, TXT):
```bash
python -m ingest --input /Users/developer/Documents/fullragimpl/docs --collection stocks_data
```
6. Run the app:
```bash
streamlit run app.py
```

## Notes
- Set `OLLAMA_MODEL` (e.g., `llama3.1:8b`) and ensure Ollama is running.
- Re-ranking uses a cross-encoder set by `RERANKER_MODEL`.
- Long-term memory is stored in Qdrant (`QDRANT_MEMORY_COLLECTION`). Short-term memory kept per-session.
- LLM Judge validates response quality with configurable threshold (`JUDGE_THRESHOLD`).

## Added LLM Judge: validates response quality, relevance, accuracy, citations, completeness, and clarity.
<img width="1024" height="376" alt="Screenshot-2025-12-06-at-11 04 18-PM-1024x376" src="https://github.com/user-attachments/assets/550e18b7-d187-43f8-80a3-e0973aa154c2" />
