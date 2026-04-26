"""Build retrieval queries from VLM structured output."""

from __future__ import annotations


def _join_items(value: object) -> str:
	"""Convert lists/strings into a compact retrieval-friendly string."""
	if isinstance(value, list):
		parts = [str(item).strip() for item in value if str(item).strip()]
		return " ".join(parts)
	if isinstance(value, str):
		return value.strip()
	return ""


def build_retrieval_query(vlm_result: dict | None, fallback_question: str | None = None) -> str:
	"""Create a dense retrieval query from VLM structured extraction output.

	Args:
		vlm_result: Expected shape from describe_frames_with_vlm.
		fallback_question: Optional user question if VLM output is missing fields.

	Returns:
		A single query string suitable for vector retrieval.
	"""
	structured = {}
	if isinstance(vlm_result, dict):
		structured = vlm_result

	question = (structured.get("question") or fallback_question or "").strip()
	play_narration = _join_items(structured.get("play_narration"))
	signals = _join_items(structured.get("query_relevant_signals"))

	# Also pick up legacy field names so callers aren't broken during transition.
	sequence_summary = _join_items(structured.get("sequence_summary"))
	evidence = _join_items(structured.get("evidence"))

	parts = [p for p in [question, play_narration, signals, sequence_summary, evidence] if p]
	query = " ".join(parts).strip()

	if not query:
		query = fallback_question or "NBA rules officiating call"

	return query
