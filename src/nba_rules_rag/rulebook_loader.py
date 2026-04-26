"""Utilities for loading and cleaning the NBA rulebook PDF."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def _clean_page_text(text: str) -> str:
	"""Normalize extracted PDF text for downstream chunking."""
	cleaned = text.replace("\x00", " ")
	cleaned = re.sub(r"\s+", " ", cleaned)
	cleaned = re.sub(r"Page\s+\d+\s+of\s+\d+", "", cleaned, flags=re.IGNORECASE)
	return cleaned.strip()


def load_rulebook_pages(pdf_path: str | Path) -> list[dict]:
	"""Load rulebook PDF pages as cleaned text records.

	Returns a list of dicts with page metadata for later chunking.
	"""
	resolved = Path(pdf_path)
	if not resolved.exists():
		raise FileNotFoundError(f"Rulebook PDF not found: {resolved}")

	reader = PdfReader(str(resolved))
	pages: list[dict] = []
	for index, page in enumerate(reader.pages, start=1):
		raw_text = page.extract_text() or ""
		text = _clean_page_text(raw_text)
		if not text:
			continue
		pages.append(
			{
				"page_num": index,
				"text": text,
				"source_path": str(resolved),
			}
		)
	if not pages:
		raise ValueError(f"No extractable text found in PDF: {resolved}")
	return pages
