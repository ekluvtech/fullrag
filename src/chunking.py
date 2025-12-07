from __future__ import annotations
from typing import List, Dict, Any, Iterable
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import html2text
from bs4 import BeautifulSoup
from pypdf import PdfReader
import docx

from config import app_config

_whitespace_re = re.compile(r"\s+")

def _normalize(text: str) -> str:
	return _whitespace_re.sub(" ", text).strip()

def load_text_from_file(path: str) -> str:
	text = ""
	extension = os.path.splitext(path)[1].lower()
	if extension in [".txt", ".md"]:
		with open(path, "r", encoding="utf-8", errors="ignore") as f:
			text = f.read()
	elif extension == ".pdf":
		reader = PdfReader(path)
		pages = []
		for page in reader.pages:
			pages.append(page.extract_text() or "")
		text = "\n".join(pages)
	elif extension in [".docx", ".doc"]:
		document = docx.Document(path)
		text = "\n".join([p.text for p in document.paragraphs])
	elif extension in [".html", ".htm"]:
		with open(path, "r", encoding="utf-8", errors="ignore") as f:
			html = f.read()
		soup = BeautifulSoup(html, "html.parser")
		for script in soup(["script", "style"]):
			script.extract()
		markdown = html2text.html2text(str(soup))
		text = markdown
	else:
		with open(path, "r", encoding="utf-8", errors="ignore") as f:
			text = f.read()
	return _normalize(text)


def recursive_directory_loader(root: str, archive_dir: str | None = None) -> List[Dict[str, Any]]:
	paths = []
	for dirpath, _, filenames in os.walk(root):
		for filename in filenames:
			full = os.path.join(dirpath, filename)
			# Skip files in archive directory
			if archive_dir and full.startswith(archive_dir):
				continue
			paths.append({"path": full})
	return paths


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	if not text:
		return []
	chunks = []
	start = 0
	length = len(text)
	while start < length:
		end = min(start + chunk_size, length)
		chunks.append(text[start:end])
		if end == length:
			break
		start = max(end - chunk_overlap, 0)
	return chunks


def make_chunks_for_path(item: Dict[str, Any]) -> List[Dict[str, Any]]:
	path = item["path"]
	text = load_text_from_file(path)
	chunks = chunk_text(text, app_config.chunk_size, app_config.chunk_overlap)
	return [{"text": c, "metadata": {"source": path, "chunk_index": i}} for i, c in enumerate(chunks)]


def archive_file(source_path: str, archive_dir: str) -> str:
	"""Move file to archive directory, preserving relative path structure"""
	# Create archive directory if it doesn't exist
	os.makedirs(archive_dir, exist_ok=True)
	
	# Get relative path from source
	rel_path = os.path.relpath(source_path)
	archive_path = os.path.join(archive_dir, rel_path)
	
	# Create archive subdirectory if needed
	archive_subdir = os.path.dirname(archive_path)
	os.makedirs(archive_subdir, exist_ok=True)
	
	# Move file to archive
	shutil.move(source_path, archive_path)
	return archive_path


def build_chunks_from_directory(root: str, archive_dir: str | None = None) -> List[Dict[str, Any]]:
	items = recursive_directory_loader(root, archive_dir)
	results: List[Dict[str, Any]] = []
	with ThreadPoolExecutor(max_workers=app_config.max_workers) as ex:
		for chunks in ex.map(make_chunks_for_path, items):
			results.extend(chunks)
	return results


def iter_chunks_from_directory(root: str, archive_dir: str | None = None) -> Iterable[Dict[str, Any]]:
	items = recursive_directory_loader(root, archive_dir)
	with ThreadPoolExecutor(max_workers=app_config.max_workers) as ex:
		futures = {ex.submit(make_chunks_for_path, it): it for it in items}
		for fut in as_completed(futures):
			for chunk in fut.result():
				yield chunk


def iter_chunks_with_archive(root: str, archive_dir: str) -> Iterable[Dict[str, Any]]:
	"""Stream chunks and archive processed files"""
	items = recursive_directory_loader(root, archive_dir)
	with ThreadPoolExecutor(max_workers=app_config.max_workers) as ex:
		futures = {ex.submit(make_chunks_for_path, it): it for it in items}
		for fut in as_completed(futures):
			item = futures[fut]
			chunks = fut.result()
			# Archive the file after processing
			try:
				archive_file(item["path"], archive_dir)
			except Exception as e:
				print(f"Warning: Failed to archive {item['path']}: {e}")
			# Yield all chunks from this file
			for chunk in chunks:
				yield chunk
