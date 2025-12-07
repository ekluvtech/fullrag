from __future__ import annotations
from typing import List, Dict, Any, Iterable
import ollama

from config import llm_config

class LLMService:
	def __init__(self):
		if llm_config.provider != "ollama":
			raise RuntimeError("Only Ollama provider is supported in current configuration")
		self.client = ollama.Client(host=llm_config.ollama_host)
		self.model = llm_config.model
		self.temperature = llm_config.temperature

	def _format_context(self, docs: List[Dict[str, Any]]) -> str:
		lines = []
		for i, d in enumerate(docs):
			meta = d.get("payload", {})
			source = meta.get("source", "unknown")
			text = meta.get("text", "")
			lines.append(f"[Doc {i+1}] Source: {source}\n{text}")
		return "\n\n".join(lines)

	def build_messages(self, query: str, short_mem: List[Dict[str, Any]], long_mem: List[Dict[str, Any]], context_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
		messages: List[Dict[str, str]] = []
		if short_mem:
			for m in short_mem[-8:]:
				messages.append({"role": m["role"], "content": m["content"]})
		if long_mem:
			snippets = [f"[{d['payload'].get('role')}] {d['payload'].get('text')}" for d in long_mem[:6]]
			messages.append({"role": "system", "content": "Relevant past memory:\n" + "\n".join(snippets)})
		if context_docs:
			messages.append({"role": "system", "content": "Use the following context to answer. Cite sources as [Doc N].\n" + self._format_context(context_docs)})
		messages.append({"role": "user", "content": query})
		return messages

	def chat(self, messages: List[Dict[str, str]]) -> Iterable[str]:
		stream = self.client.chat(model=self.model, messages=messages, options={"temperature": self.temperature}, stream=True)
		for chunk in stream:
			yield chunk.get("message", {}).get("content", "")
