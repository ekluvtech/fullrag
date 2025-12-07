import os
import streamlit as st

from config import app_config, judge_config
from retrieval import Retriever
from memory import ShortTermMemory, LongTermMemory
from llm import LLMService
from judge import LLMJudge
from vectorstore import QdrantStore, init_default_collections

st.set_page_config(page_title=app_config.app_title, layout="wide")

if "short_mem" not in st.session_state:
	st.session_state.short_mem = ShortTermMemory(max_messages=30)
if "session_id" not in st.session_state:
	st.session_state.session_id = os.getenv("SESSION_ID", "default-session")

st.title(app_config.app_title)

with st.sidebar:
	st.header("Settings")
	top_k = st.slider("Top K", 5, 50, 20)
	mmr_k = st.slider("MMR K", 2, 20, 8)
	use_memory = st.checkbox("Use long-term memory", value=True)
	enable_judge = st.checkbox("Enable LLM Judge", value=judge_config.enabled)
	judge_threshold = st.slider("Judge Threshold", 1.0, 10.0, judge_config.threshold, 0.5)
	st.divider()
	#st.markdown("Start Qdrant via: `docker compose up -d qdrant`")

store = QdrantStore()
init_default_collections(store)
retriever = Retriever(store=store)
mem_long = LongTermMemory(store=store)
llm = None
judge = None
try:
	llm = LLMService()
	judge = LLMJudge()
except Exception as e:
	st.warning(f"LLM not initialized: {e}")

chat_container = st.container()

with st.form(key="chat_form", clear_on_submit=True):
	query = st.text_area("Ask a question", height=120)
	submitted = st.form_submit_button("Send")

if submitted and query.strip():
	st.session_state.short_mem.add("user", query)
	long_mem_docs = mem_long.recall(st.session_state.session_id, query, top_k=5) if use_memory else []
	docs = retriever.search(query, top_k=top_k, mmr_k=mmr_k)
	with chat_container:
		st.markdown("### Answer")
		if llm:
			messages = llm.build_messages(query, st.session_state.short_mem.get(), long_mem_docs, docs)
			placeholder = st.empty()
			col1, col2 = st.columns([2,1])
			with col1:
				accum = ""
				for token in llm.chat(messages):
					accum += token
					placeholder.markdown(accum)
				
				# Judge validation if enabled
				if enable_judge and judge:
					with st.spinner("Validating response..."):
						judgment = judge.validate_response(query, accum, docs)
						should_regenerate = judge.should_regenerate(judgment, judge_threshold)
						
						if should_regenerate:
							st.warning("⚠️ Response quality below threshold. Consider regenerating.")
							st.markdown("**Judge Evaluation:**")
							st.json(judgment)
						else:
							st.success("✅ Response quality validated")
							with st.expander("Judge Details"):
								st.json(judgment)
				
				st.session_state.short_mem.add("assistant", accum)
				if use_memory:
					mem_long.add(st.session_state.session_id, "assistant", accum)
			with col2:
				st.markdown("### Sources")
				for i, d in enumerate(docs):
					meta = d.get("payload", {})
					source = meta.get("source", "unknown")
					st.write(f"[Doc {i+1}] {source}")
		else:
			st.info("Provide OLLAMA in .env to enable answers.")
