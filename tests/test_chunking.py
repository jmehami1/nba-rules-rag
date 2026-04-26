from __future__ import annotations

from nba_rules_rag.chunking import chunk_rule_sections, split_rule_sections


def test_split_rule_sections_detects_rule_headers():
	pages = [
		{
			"page_num": 1,
			"source_path": "dummy.pdf",
			"text": "RULE 4 - Definitions Section text for definitions.",
		},
		{
			"page_num": 2,
			"source_path": "dummy.pdf",
			"text": "RULE 10 - Violations Traveling occurs when a player...",
		},
	]

	sections = split_rule_sections(pages)

	assert len(sections) == 2
	assert sections[0]["rule_id"] == "4"
	assert sections[1]["rule_id"] == "10"


def test_chunk_rule_sections_creates_metadata_chunks():
	sections = [
		{
			"rule_id": "10",
			"section_title": "RULE 10 - Violations",
			"page_start": 12,
			"page_end": 13,
			"source_path": "dummy.pdf",
			"text": "word " * 250,
		}
	]

	chunks = chunk_rule_sections(sections, chunk_size_words=100, overlap_words=20)

	assert len(chunks) >= 2
	assert chunks[0]["rule_id"] == "10"
	assert chunks[0]["page_start"] == 12
	assert chunks[0]["source_path"] == "dummy.pdf"
