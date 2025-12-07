from __future__ import annotations
from typing import List, Dict, Any
import json
import ollama
from config import llm_config

class LLMJudge:
	def __init__(self, model_name: str | None = None):
		self.client = ollama.Client(host=llm_config.ollama_host)
		self.model = model_name or llm_config.model

	def validate_response(self, query: str, response: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Validate response quality, relevance, and citation accuracy"""
		
		# Prepare context for judge
		context_text = ""
		for i, doc in enumerate(context_docs):
			meta = doc.get("payload", {})
			source = meta.get("source", "unknown")
			text = meta.get("text", "")
			context_text += f"[Doc {i+1}] Source: {source}\n{text}\n\n"
		
		judge_prompt = f"""
You are an expert judge evaluating RAG responses. Rate the following response on multiple criteria.

Query: {query}

Context Documents:
{context_text}

Response to evaluate:
{response}

Please evaluate and provide scores (1-10) for:
1. Relevance: How well does the response answer the query?
2. Accuracy: Is the information factually correct based on context?
3. Citation Quality: Are sources properly cited and relevant?
4. Completeness: Does the response fully address the query?
5. Clarity: Is the response clear and well-structured?

Provide your evaluation in JSON format:
{{
    "relevance_score": <1-10>,
    "accuracy_score": <1-10>,
    "citation_score": <1-10>,
    "completeness_score": <1-10>,
    "clarity_score": <1-10>,
    "overall_score": <1-10>,
    "issues": ["list of specific issues found"],
    "recommendations": ["suggestions for improvement"]
}}
"""
		
		try:
			response_obj = self.client.chat(
				model=self.model,
				messages=[{"role": "user", "content": judge_prompt}],
				options={"temperature": 0.1}  # Low temperature for consistent evaluation
			)
			
			judge_text = response_obj["message"]["content"]
			
			# Try to parse JSON response
			try:
				# Extract JSON from response (handle cases where judge adds extra text)
				start_idx = judge_text.find('{')
				end_idx = judge_text.rfind('}') + 1
				if start_idx != -1 and end_idx > start_idx:
					json_text = judge_text[start_idx:end_idx]
					judgment = json.loads(json_text)
				else:
					raise ValueError("No JSON found in response")
			except (json.JSONDecodeError, ValueError):
				# Fallback if JSON parsing fails
				judgment = {
					"relevance_score": 5,
					"accuracy_score": 5,
					"citation_score": 5,
					"completeness_score": 5,
					"clarity_score": 5,
					"overall_score": 5,
					"issues": ["Failed to parse judge response"],
					"recommendations": ["Check judge model response format"]
				}
			
			return judgment
			
		except Exception as e:
			return {
				"relevance_score": 5,
				"accuracy_score": 5,
				"citation_score": 5,
				"completeness_score": 5,
				"completeness_score": 5,
				"clarity_score": 5,
				"overall_score": 5,
				"issues": [f"Judge evaluation failed: {str(e)}"],
				"recommendations": ["Check judge service configuration"]
			}

	def should_regenerate(self, judgment: Dict[str, Any], threshold: float = 6.0) -> bool:
		"""Determine if response should be regenerated based on scores"""
		overall_score = judgment.get("overall_score", 5)
		return overall_score < threshold
