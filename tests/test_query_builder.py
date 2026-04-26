from __future__ import annotations

import json

import numpy as np

from nba_rules_rag.embeddings import build_faiss_index, save_rulebook_index
from nba_rules_rag.query_builder import build_retrieval_query
from nba_rules_rag.retriever import load_rulebook_index, retrieve_rulebook_chunks


def test_build_faiss_index_and_save(tmp_path):
	chunks = [
		{
			"chunk_id": "rule_10_0",
			"rule_id": "10",
			"section_title": "RULE 10 - Violations",
			"page_start": 12,
			"page_end": 12,
			"source_path": "dummy.pdf",
			"text": "Traveling occurs when a player moves illegally.",
		}
	]
	embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

	index = build_faiss_index(embeddings)
	saved = save_rulebook_index(chunks, embeddings, index, tmp_path)

	assert (tmp_path / "chunks.jsonl").exists()
	assert (tmp_path / "embeddings.npy").exists()
	assert (tmp_path / "faiss.index").exists()
	metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
	assert metadata["num_chunks"] == 1
	assert saved["index_path"].endswith("faiss.index")


def test_retrieve_rulebook_chunks_from_saved_store(tmp_path, monkeypatch):
	chunks = [
		{
			"chunk_id": "rule_10_0",
			"rule_id": "10",
			"section_title": "RULE 10 - Violations",
			"page_start": 12,
			"page_end": 12,
			"source_path": "dummy.pdf",
			"text": "Traveling occurs when a player moves illegally while holding the ball.",
		},
		{
			"chunk_id": "rule_4_0",
			"rule_id": "4",
			"section_title": "RULE 4 - Definitions",
			"page_start": 3,
			"page_end": 3,
			"source_path": "dummy.pdf",
			"text": "The gather is the point where a player gains control of the ball.",
		},
	]
	embeddings = np.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
		],
		dtype=np.float32,
	)
	index = build_faiss_index(embeddings)
	save_rulebook_index(chunks, embeddings, index, tmp_path)

	def _fake_embed_texts(texts, model_name="unused"):
		assert texts == ["travel rule"]
		return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

	monkeypatch.setattr("nba_rules_rag.retriever.embed_texts", _fake_embed_texts)

	loaded_index, loaded_chunks = load_rulebook_index(tmp_path)
	assert loaded_index.ntotal == 2
	assert len(loaded_chunks) == 2

	results = retrieve_rulebook_chunks("travel rule", top_k=2, vector_store_dir=tmp_path)
	assert len(results) == 2
	assert results[0]["rule_id"] == "10"
	assert results[0]["score"] >= results[1]["score"]


def test_build_retrieval_query_from_vlm_structured_output():
	vlm_result = {
		"question": "Was this traveling?",
		"play_narration": "Player gathers and moves left.",
		"query_relevant_signals": ["Ball secured near torso", "No visible dribble after gather"],
	}

	query = build_retrieval_query(vlm_result)

	assert "Was this traveling?" in query
	assert "Player gathers and moves left." in query
	assert "Ball secured near torso" in query


def test_build_retrieval_query_with_fallback_question_only():
	query = build_retrieval_query(None, fallback_question="Did the player travel?")

	assert "Did the player travel?" in query
