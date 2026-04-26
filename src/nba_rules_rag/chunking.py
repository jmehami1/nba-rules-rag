"""Chunking helpers for NBA rulebook text."""

from __future__ import annotations

import re


RULE_HEADER_RE = re.compile(r"(RULE\s+\d+[A-Z]?(?:\s*[\-–:]\s*[^R]+)?)", re.IGNORECASE)


def split_rule_sections(pages: list[dict]) -> list[dict]:
	"""Split cleaned page text into coarse rule sections using RULE headers."""
	combined_parts: list[str] = []
	page_markers: list[tuple[int, int]] = []
	cursor = 0
	for page in pages:
		marker = f"\n[[PAGE:{page['page_num']}]]\n"
		part = marker + page["text"]
		combined_parts.append(part)
		page_markers.append((cursor, page["page_num"]))
		cursor += len(part)

	combined_text = "".join(combined_parts)
	matches = list(RULE_HEADER_RE.finditer(combined_text))
	if not matches:
		return [
			{
				"rule_id": "unknown",
				"section_title": "Full Rulebook",
				"text": combined_text.replace("\n", " ").strip(),
				"page_start": pages[0]["page_num"],
				"page_end": pages[-1]["page_num"],
				"source_path": pages[0]["source_path"],
			}
		]

	sections: list[dict] = []
	for idx, match in enumerate(matches):
		start = match.start()
		end = matches[idx + 1].start() if idx + 1 < len(matches) else len(combined_text)
		raw_text = combined_text[start:end].strip()
		title = match.group(1).strip()
		rule_num_match = re.search(r"RULE\s+(\d+[A-Z]?)", title, re.IGNORECASE)
		rule_id = rule_num_match.group(1) if rule_num_match else "unknown"
		page_nums = [int(p) for p in re.findall(r"\[\[PAGE:(\d+)\]\]", raw_text)]
		section_text = re.sub(r"\[\[PAGE:\d+\]\]", "", raw_text)
		section_text = re.sub(r"\s+", " ", section_text).strip()
		sections.append(
			{
				"rule_id": rule_id,
				"section_title": title,
				"text": section_text,
				"page_start": min(page_nums) if page_nums else pages[0]["page_num"],
				"page_end": max(page_nums) if page_nums else pages[-1]["page_num"],
				"source_path": pages[0]["source_path"],
			}
		)
	return sections


def chunk_rule_sections(sections: list[dict], chunk_size_words: int = 180, overlap_words: int = 40) -> list[dict]:
	"""Chunk rule sections into embedding-sized windows with metadata."""
	if chunk_size_words <= 0:
		raise ValueError("chunk_size_words must be positive")
	if overlap_words < 0:
		raise ValueError("overlap_words must be non-negative")
	if overlap_words >= chunk_size_words:
		raise ValueError("overlap_words must be smaller than chunk_size_words")

	chunks: list[dict] = []
	for section in sections:
		words = section["text"].split()
		if not words:
			continue
		start = 0
		chunk_index = 0
		while start < len(words):
			end = min(len(words), start + chunk_size_words)
			text = " ".join(words[start:end]).strip()
			if text:
				chunks.append(
					{
						"chunk_id": f"rule_{section['rule_id']}_{chunk_index}",
						"rule_id": section["rule_id"],
						"section_title": section["section_title"],
						"page_start": section["page_start"],
						"page_end": section["page_end"],
						"source_path": section["source_path"],
						"text": text,
					}
				)
			if end >= len(words):
				break
			start = end - overlap_words
			chunk_index += 1
	return chunks
