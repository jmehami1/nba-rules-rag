"""Retriever utilities for FAISS-backed NBA rulebook search."""

from __future__ import annotations

import json
from pathlib import Path

import faiss

from nba_rules_rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts


DEFAULT_VECTOR_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "vector_store"


def load_rulebook_chunks(chunks_path: str | Path) -> list[dict]:
	"""Load chunk metadata from a JSONL file."""
	resolved = Path(chunks_path)
	if not resolved.exists():
		raise FileNotFoundError(f"Chunk metadata file not found: {resolved}")

	chunks: list[dict] = []
	with resolved.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			chunks.append(json.loads(line))
	if not chunks:
		raise ValueError(f"No chunks found in metadata file: {resolved}")
	return chunks


def load_rulebook_index(vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR) -> tuple[faiss.Index, list[dict]]:
	"""Load FAISS index and aligned chunk metadata from a vector store directory."""
	resolved = Path(vector_store_dir)
	index_path = resolved / "faiss.index"
	chunks_path = resolved / "chunks.jsonl"
	if not index_path.exists():
		raise FileNotFoundError(f"FAISS index not found: {index_path}")

	index = faiss.read_index(str(index_path))
	chunks = load_rulebook_chunks(chunks_path)
	if index.ntotal != len(chunks):
		raise ValueError(
			f"Index/chunk mismatch: index has {index.ntotal} vectors, chunks file has {len(chunks)} rows"
		)
	return index, chunks


def retrieve_rulebook_chunks(
	query: str,
	top_k: int = 5,
	vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
	model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
	"""Return top-k retrieved rulebook chunks for a natural-language query."""
	clean_query = (query or "").strip()
	if not clean_query:
		raise ValueError("query must be a non-empty string")
	if top_k <= 0:
		raise ValueError("top_k must be positive")

	index, chunks = load_rulebook_index(vector_store_dir)
	query_embedding = embed_texts([clean_query], model_name=model_name)
	k = min(top_k, len(chunks))
	scores, indices = index.search(query_embedding, k)

	results: list[dict] = []
	for score, chunk_idx in zip(scores[0], indices[0]):
		if chunk_idx < 0:
			continue
		chunk = dict(chunks[int(chunk_idx)])
		chunk["score"] = float(score)
		results.append(chunk)
	return results
