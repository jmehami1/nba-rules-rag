"""Embedding and vector-index helpers for NBA rulebook chunks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Suppress verbose load-report output from sentence-transformers / transformers.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Strong, modern, and Colab-friendly embedding model for English retrieval.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
	"""Return a cached SentenceTransformer, loading it once per process."""
	if model_name not in _model_cache:
		device = os.getenv("EMBEDDING_DEVICE", "cpu")
		_model_cache[model_name] = SentenceTransformer(model_name, device=device)
	return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
	"""Embed text chunks using a sentence-transformers model."""
	if not texts:
		raise ValueError("texts must contain at least one string")
	model = _get_model(model_name)
	embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
	return np.asarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
	"""Build an inner-product FAISS index over normalized embeddings."""
	if embeddings.ndim != 2 or embeddings.shape[0] == 0:
		raise ValueError("embeddings must be a non-empty 2D array")
	dim = int(embeddings.shape[1])
	index = faiss.IndexFlatIP(dim)
	index.add(embeddings)
	return index


def save_rulebook_index(chunks: list[dict], embeddings: np.ndarray, index: faiss.Index, output_dir: str | Path) -> dict:
	"""Persist chunk metadata, embeddings, and FAISS index to disk."""
	resolved = Path(output_dir)
	resolved.mkdir(parents=True, exist_ok=True)

	chunks_path = resolved / "chunks.jsonl"
	embeddings_path = resolved / "embeddings.npy"
	index_path = resolved / "faiss.index"
	metadata_path = resolved / "metadata.json"

	with chunks_path.open("w", encoding="utf-8") as handle:
		for chunk in chunks:
			handle.write(json.dumps(chunk, ensure_ascii=True) + "\n")

	np.save(embeddings_path, embeddings)
	faiss.write_index(index, str(index_path))

	metadata = {
		"num_chunks": len(chunks),
		"embedding_dim": int(embeddings.shape[1]),
		"index_type": type(index).__name__,
	}
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	return {
		"chunks_path": str(chunks_path),
		"embeddings_path": str(embeddings_path),
		"index_path": str(index_path),
		"metadata_path": str(metadata_path),
	}
