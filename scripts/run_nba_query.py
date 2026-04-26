"""CLI runner for NBA query analysis on a YouTube clip."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Suppress noisy HuggingFace Hub and transformers logs before any imports load them.
os.environ.setdefault("HF_HUB_VERBOSITY", "warning")
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Ensure local execution works even without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from nba_rules_rag.frame_extraction import extract_frames_from_youtube
from nba_rules_rag.rulebook_loader import load_rulebook_pages
from nba_rules_rag.chunking import chunk_rule_sections, split_rule_sections
from nba_rules_rag.embeddings import build_faiss_index, embed_texts, save_rulebook_index
from nba_rules_rag.vlm_describer import build_vlm_user_prompt, describe_frames_with_vlm
from nba_rules_rag.youtube_utils import parse_youtube_url, validate_interval
from nba_rules_rag.query_builder import build_retrieval_query
from nba_rules_rag.retriever import retrieve_rulebook_chunks

DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=9fFWawcJXUw"
DEFAULT_START_TIME = "0:21"
DEFAULT_END_TIME = "0:29"
DEFAULT_QUESTION = "Was this player in blue uniform travelling?"
DEFAULT_RULEBOOK_PDF_PATH = str(REPO_ROOT / "docs" / "2023-24-NBA-Season-Official-Playing-Rules.pdf")
DEFAULT_VECTOR_STORE_DIR = str(REPO_ROOT / "data" / "vector_store")

DEMO_RULING = {
	"question": "Was this player in blue uniform travelling?",
	"applicable_rules": [
		{
			"rule": "Rule 4, Section III(b)",
			"application": (
				"For a dribbler, the gather occurs when the player puts two hands on the ball, "
				"lets the ball come to rest, puts a hand under the ball and pauses it, or otherwise "
				"gains enough control to hold, pass, shoot, change hands, or cradle the ball. "
				"Based on the play description, F03 is the most likely gather frame because the ball "
				"appears to come into two-hand control near the player's chest."
			),
		},
		{
			"rule": "Rule 8, Section XIII(b)",
			"application": (
				"After gathering while dribbling, a progressing player may take two steps in coming "
				"to a stop, passing, or shooting. The first step is the first foot or both feet touching "
				"the floor after the gather, and the second step is the other foot touching after that "
				"or both feet touching simultaneously. The provided frames do not clearly establish the "
				"exact post-gather foot contacts, so the two-step rule cannot be conclusively applied "
				"to find a violation."
			),
		},
	],
	"likely_gather_frame": "F03",
	"gather_confidence": "medium",
	"post_gather_step_count_visible": "unclear",
	"pivot_foot": "unclear",
	"travel_ruling": "inconclusive",
	"ruling_confidence": "medium",
	"reasoning": (
		"The player appears to gather around F03 when the ball comes into two-hand control. "
		"However, the sampled frames do not clearly show the first and second foot contacts after "
		"that gather, nor do they establish a pivot foot being lifted and returned to the floor. "
		"The player changes posture and appears to come to a stop while holding the ball, but "
		"without continuous video or clearly visible footfall timing, there is not enough visible "
		"evidence to rule this a travelling violation under the NBA travelling rule."
	),
	"facts_required_for_travel_call": [
		"The exact moment the ball was gathered or came to rest.",
		"Which foot, if any, was on the floor at the gather.",
		"The first foot or feet to touch the floor after the gather.",
		"The second foot or feet to touch the floor after the gather.",
		"Whether the player touched the floor consecutively with the same foot after ending the dribble.",
		"Whether a pivot foot was established, lifted, and returned to the floor before a pass, shot, or dribble release.",
	],
	"limitations": [
		"The frames are sampled approximately 1.33 seconds apart, so important foot contacts between frames may be missing.",
		"The exact gather frame is likely F03 but not certain from the still images alone.",
		"The player's foot placement is not clear enough in every frame to identify a pivot foot.",
		"The sequence does not clearly show a jump-and-land action while the player holds the ball.",
		"Because the required post-gather footfall sequence is ambiguous, a definitive travel ruling cannot be made from the provided frames.",
	],
}


def ensure_rulebook_embeddings(
	rulebook_pdf_path: str,
	vector_store_dir: str,
	chunk_size_words: int = 180,
	overlap_words: int = 40,
) -> dict:
	"""Ensure rulebook embeddings/index exist, building them when missing."""
	store_dir = Path(vector_store_dir)
	index_path = store_dir / "faiss.index"
	chunks_path = store_dir / "chunks.jsonl"
	metadata_path = store_dir / "metadata.json"

	if index_path.exists() and chunks_path.exists() and metadata_path.exists():
		return {
			"vector_store_dir": str(store_dir),
			"built": False,
			"pdf_path": rulebook_pdf_path,
			"index_path": str(index_path),
			"chunks_path": str(chunks_path),
			"metadata_path": str(metadata_path),
		}

	pages = load_rulebook_pages(rulebook_pdf_path)
	sections = split_rule_sections(pages)
	chunks = chunk_rule_sections(
		sections,
		chunk_size_words=chunk_size_words,
		overlap_words=overlap_words,
	)
	texts = [chunk["text"] for chunk in chunks]
	chunk_embeddings = embed_texts(texts)
	index = build_faiss_index(chunk_embeddings)
	saved = save_rulebook_index(chunks, chunk_embeddings, index, store_dir)
	return {
		"vector_store_dir": str(store_dir),
		"built": True,
			"pdf_path": rulebook_pdf_path,
		**saved,
	}


def build_vlm_output_embedding(vlm_result: dict, output_dir: str | Path) -> dict:
	"""Build and persist an embedding for structured VLM output."""
	if not isinstance(vlm_result, dict) or not vlm_result:
		raise ValueError("vlm_result must be a non-empty dict")

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	vlm_text = json.dumps(vlm_result, ensure_ascii=True, sort_keys=True)
	vlm_embedding = embed_texts([vlm_text])

	embedding_path = output_path / "vlm_output_embedding.npy"
	text_path = output_path / "vlm_output_text.json"
	np.save(embedding_path, vlm_embedding)
	text_path.write_text(vlm_text, encoding="utf-8")

	return {
		"vlm_embedding_path": str(embedding_path),
		"vlm_embedding_text_path": str(text_path),
		"vlm_embedding_dim": int(vlm_embedding.shape[1]),
	}


def build_mock_vlm_result(question: str, n_frames: int) -> dict:
	"""Return a deterministic mock VLM output for local development."""
	_ = n_frames
	return {
		"question": question,
		"play_narration": (
			"The player in blue moves left toward the sideline while controlling the ball low with "
			"the left hand in the opening frames. Across F01 to F02, the ball appears consistent with "
			"an active dribble or the end of a dribble. By F03, the player appears to bring the ball "
			"into two-hand control near the chest, and from F03 through F07 the ball remains held "
			"rather than visibly dribbled. After that apparent gather, the player continues to lean "
			"and reposition near the sideline, then settles into a stationary stance facing the defender. "
			"This sequence is relevant to the travelling question because the play appears to transition "
			"from dribble to held ball, but the sampled frames do not clearly show the exact gather "
			"moment or every foot contact needed to determine a definitive travelling violation from the "
			"images alone."
		),
		"frame_observations": [
			{
				"frame_id": "F01",
				"timestamp_sec": 21.0,
				"description": (
					"The blue player is moving leftward near the sideline with the ball low on the left side "
					"of the body. The torso is leaning forward. The defender in red is in front and slightly "
					"to the right. The exact foot planted is not fully clear, but the player is in motion."
				),
				"change_from_previous": "N/A — first frame",
			},
			{
				"frame_id": "F02",
				"timestamp_sec": 22.33,
				"description": (
					"The blue player continues moving left. The ball is still on the left side and not yet "
					"clearly secured with two hands. The left leg is extended forward/left, and the right leg "
					"trails behind. The defender shifts laterally to stay in front."
				),
				"change_from_previous": (
					"The player advances left and the ball rises slightly from the very low control position "
					"seen in F01."
				),
			},
			{
				"frame_id": "F03",
				"timestamp_sec": 23.67,
				"description": (
					"The blue player appears to have both hands on the ball near the chest/left side, "
					"suggesting the ball is now gathered or being gathered. The body remains angled left, "
					"with the left leg forward and the right leg behind. The defender is still squared up "
					"in front."
				),
				"change_from_previous": (
					"The ball transitions from a left-side dribble/control position to apparent two-hand "
					"control near the torso."
				),
			},
			{
				"frame_id": "F04",
				"timestamp_sec": 25.0,
				"description": (
					"The blue player bends lower and tucks the ball tightly with both hands near the upper "
					"torso. The left foot appears to be the more forward foot near the sideline, while the "
					"right leg trails behind. No visible dribble is present."
				),
				"change_from_previous": (
					"The player lowers the torso further and continues holding the ball securely rather than "
					"appearing to dribble."
				),
			},
			{
				"frame_id": "F05",
				"timestamp_sec": 26.33,
				"description": (
					"The blue player remains crouched, still holding the ball tightly with both hands. The "
					"player is close to the sideline and appears to be stabilizing balance. The defender is "
					"slightly back on the heels and still in front."
				),
				"change_from_previous": (
					"The player stays in two-hand possession and lowers even more, with no visible bounce "
					"or release of the ball."
				),
			},
			{
				"frame_id": "F06",
				"timestamp_sec": 27.67,
				"description": (
					"The blue player rises somewhat from the crouch and holds the ball lower in front of "
					"the body. Both feet are visible and separated, with the left foot nearer the sideline "
					"and the right foot farther inside the court. The player faces the defender."
				),
				"change_from_previous": (
					"The player comes up from the deep crouch into a more balanced stance while still "
					"holding the ball."
				),
			},
			{
				"frame_id": "F07",
				"timestamp_sec": 29.0,
				"description": (
					"The blue player remains in possession of the ball, facing the defender in a more set "
					"stance. The feet are still apart, and the body is more upright than in prior frames. "
					"No shot, pass, or renewed dribble is visible."
				),
				"change_from_previous": (
					"The player becomes more upright and stationary while maintaining control of the ball."
				),
			},
		],
		"query_relevant_signals": [
			"The ball appears low and left-sided in F01-F02, consistent with a live dribble or the end of a dribble.",
			"By F03, the ball appears to be in two-hand control near the chest, indicating a likely gather point or immediate post-gather state.",
			"From F03 through F07, no visible dribble resumes; the ball remains held.",
			"The player continues changing body position after the apparent gather, including leaning, crouching, and then rising into a set stance.",
			"A clear pivot foot cannot be reliably identified from the provided frames.",
			"The frames do not clearly show a jump-and-land sequence after the apparent gather.",
		],
		"uncertainties": [
			"The exact instant of gather is not fully visible because the frames are sampled and not continuous video.",
			"Some foot contacts between frames may be missing, so a precise post-gather step count cannot be confirmed from these images alone.",
			"The player's feet are partially obscured or not crisp enough in some frames to identify a definitive pivot foot.",
			"It is unclear whether any subtle foot repositioning occurs between frames F03-F06 that would be material to a travelling determination.",
			"Because the sequence is sampled approximately 1.33 seconds apart, important intermediate motion is not shown.",
		],
	}


def estimate_frame_timestamps(start_sec: float, end_sec: float, n_frames: int) -> list[float]:
	"""Estimate per-frame timestamps across the selected interval."""
	if n_frames <= 0:
		return []
	if n_frames == 1:
		return [start_sec]
	step = (end_sec - start_sec) / (n_frames - 1)
	return [start_sec + i * step for i in range(n_frames)]


def build_parser() -> argparse.ArgumentParser:
	"""Build CLI argument parser with stable development defaults."""
	parser = argparse.ArgumentParser(
		description="Extract 5-10 keyframes from a YouTube clip interval."
	)
	parser.add_argument(
		"--youtube-url",
		default=DEFAULT_YOUTUBE_URL,
		help="YouTube URL to analyze.",
	)
	parser.add_argument(
		"--start-time",
		default=DEFAULT_START_TIME,
		help="Start timestamp (SS, MM:SS, or HH:MM:SS).",
	)
	parser.add_argument(
		"--end-time",
		default=DEFAULT_END_TIME,
		help="End timestamp (SS, MM:SS, or HH:MM:SS).",
	)
	parser.add_argument(
		"--question",
		default=DEFAULT_QUESTION,
		help="Question/query about the play.",
	)
	parser.add_argument(
		"--n-frames",
		type=int,
		default=8,
		help="Target frame count (will be clamped to 5-10 by extractor).",
	)
	parser.add_argument(
		"--output-dir",
		default=str(REPO_ROOT / "data" / "processed" / "demo_frames"),
		help="Directory where extracted frames and preview grid are saved.",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Disable interactive matplotlib display.",
	)
	parser.add_argument(
		"--demo",
		action="store_true",
		help=(
			"Run the built-in default demo example and use saved VLM output "
			"without any live LLM/VLM API calls."
		),
	)
	parser.add_argument(
		"--rulebook-pdf-path",
		default=DEFAULT_RULEBOOK_PDF_PATH,
		help="Path to the NBA rulebook PDF used to build embeddings if missing.",
	)
	parser.add_argument(
		"--vector-store-dir",
		default=DEFAULT_VECTOR_STORE_DIR,
		help="Directory containing FAISS index and rulebook chunk embeddings.",
	)
	return parser


def validate_question(question: str) -> str:
	"""Validate question/query input and return cleaned text."""
	cleaned = (question or "").strip()
	if not cleaned:
		raise ValueError("Question/query must be a non-empty string.")
	return cleaned


def save_frame_grid(frames, output_path: Path) -> Path:
	"""Save a simple grid preview image for extracted frames."""
	n = len(frames)
	cols = min(4, n)
	rows = math.ceil(n / cols)

	fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
	if rows == 1 and cols == 1:
		axes = [axes]
	elif rows == 1 or cols == 1:
		axes = list(axes)
	else:
		axes = [ax for row in axes for ax in row]

	for idx, ax in enumerate(axes):
		if idx < n:
			ax.imshow(frames[idx])
			ax.set_title(f"Frame {idx + 1}")
		ax.axis("off")

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=160)
	plt.close(fig)
	return output_path


def load_demo_frames(output_dir: Path) -> list:
	"""Load saved demo frames from disk in sorted order."""
	from PIL import Image
	paths = sorted(output_dir.glob("frame_*.jpg"))
	if not paths:
		raise FileNotFoundError(f"No demo frames found in {output_dir}. Run without --demo first to extract frames.")
	return [Image.open(p) for p in paths]


def run_nba_query(args: argparse.Namespace) -> dict:
	"""Run validated frame extraction and optional demo-mode narration."""
	rulebook_store_status = ensure_rulebook_embeddings(
		rulebook_pdf_path=args.rulebook_pdf_path,
		vector_store_dir=args.vector_store_dir,
	)

	youtube_url = DEFAULT_YOUTUBE_URL if args.demo else args.youtube_url
	start_time = DEFAULT_START_TIME if args.demo else args.start_time
	end_time = DEFAULT_END_TIME if args.demo else args.end_time
	question = DEFAULT_QUESTION if args.demo else args.question

	video_id = parse_youtube_url(youtube_url)
	start_sec, end_sec = validate_interval(start_time, end_time)
	cleaned_question = validate_question(question)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# In demo mode, load saved frames from disk; otherwise extract from YouTube
	if args.demo:
		frames = load_demo_frames(output_dir)
	else:
		frames = extract_frames_from_youtube(
			video_id=video_id,
			start_sec=start_sec,
			end_sec=end_sec,
			n_frames=args.n_frames,
			output_dir=str(output_dir),
		)
		for i, frame in enumerate(frames, start=1):
			frame.save(output_dir / f"frame_{i:02d}.jpg")

	grid_path = save_frame_grid(frames, output_dir / "frame_grid.jpg")
	frame_timestamps_sec = estimate_frame_timestamps(start_sec, end_sec, len(frames))
	vlm_prompt = build_vlm_user_prompt(cleaned_question, frame_timestamps_sec)

	vlm_result = None
	vlm_error = None
	vlm_embedding = None
	vlm_embedding_error = None
	if args.demo:
		vlm_result = build_mock_vlm_result(cleaned_question, len(frames))
	else:
		try:
			vlm_result = describe_frames_with_vlm(
				frames=frames,
				question=cleaned_question,
				frame_timestamps_sec=frame_timestamps_sec,
			)
		except (ValueError, RuntimeError) as exc:
			vlm_error = str(exc)

	if vlm_result is not None:
		try:
			vlm_embedding = build_vlm_output_embedding(vlm_result, output_dir)
		except (ValueError, RuntimeError) as exc:
			vlm_embedding_error = str(exc)

	rag_query = build_retrieval_query(vlm_result, cleaned_question)
	rag_chunks: list[dict] | None = None
	rag_error: str | None = None
	try:
		rag_chunks = retrieve_rulebook_chunks(
			query=rag_query,
			top_k=5,
			vector_store_dir=args.vector_store_dir,
		)
	except (ValueError, RuntimeError, FileNotFoundError) as exc:
		rag_error = str(exc)

	if not args.no_show:
		plt.figure(figsize=(12, 8))
		plt.imshow(plt.imread(grid_path))
		plt.title("Extracted Keyframes")
		plt.axis("off")
		plt.show()

	result = {
		"youtube_url": youtube_url,
		"video_id": video_id,
		"start_time": start_time,
		"end_time": end_time,
		"start_sec": start_sec,
		"end_sec": end_sec,
		"question": cleaned_question,
		"n_frames": len(frames),
		"frame_timestamps_sec": frame_timestamps_sec,
		"output_dir": str(output_dir),
		"grid_path": str(grid_path),
		"rulebook_embedding_store": rulebook_store_status,
		"vlm_prompt": vlm_prompt,
		"vlm_result": vlm_result,
		"vlm_error": vlm_error,
		"vlm_embedding": vlm_embedding,
		"vlm_embedding_error": vlm_embedding_error,
		"rag_query": rag_query,
		"rag_chunks": rag_chunks,
		"rag_error": rag_error,
		"ruling": DEMO_RULING if args.demo else None,
	}
	return result


def main() -> int:
	"""CLI entry point."""
	parser = build_parser()
	if len(sys.argv) == 1:
		parser.print_help()
		return 0
	args = parser.parse_args()
	try:
		result = run_nba_query(args)
	except ValueError as exc:
		print(f"Input validation error: {exc}", file=sys.stderr)
		return 2
	except RuntimeError as exc:
		print(f"Processing error: {exc}", file=sys.stderr)
		return 3

	sep = "═" * 72
	thin = "─" * 72

	# ── Header ──────────────────────────────────────────────────────────
	print(sep)
	print(" NBA Rules RAG")
	print(sep)
	print(f"  YouTube URL : {result['youtube_url']}")
	print(f"  Video ID    : {result['video_id']}")
	print(
		f"  Interval    : {result['start_time']} → {result['end_time']} "
		f"({result['end_sec'] - result['start_sec']:.2f}s)"
	)
	print(f"  Question    : {result['question']}")
	print(f"  Frames      : {result['n_frames']}")
	print(f"  Timestamps  : {result['frame_timestamps_sec']}")
	print(f"  Output dir  : {result['output_dir']}")
	print(f"  Grid preview: {result['grid_path']}")

	# ── Rulebook Embeddings ──────────────────────────────────────────────
	print()
	print(sep)
	print(" RULEBOOK EMBEDDINGS")
	print(sep)
	store = result["rulebook_embedding_store"]
	if store["built"]:
		print("  Status : Built in this run")
		print(f"  PDF    : {store['pdf_path']}")
		print(f"  Chunks : {store['num_chunks']} (from {store['num_pages']} pages, {store['num_sections']} sections)")
	else:
		print("  Status : Already present — skipped build")
		print(f"  Index  : {store['index_path']}")
	print(f"  Store  : {store['vector_store_dir']}")

	# ── VLM Prompt ───────────────────────────────────────────────────────
	print()
	print(sep)
	print(" VLM PROMPT")
	print(sep)
	print(result["vlm_prompt"])

	# ── VLM Output ───────────────────────────────────────────────────────
	print()
	print(sep)
	print(" VLM OUTPUT")
	print(sep)
	if result.get("vlm_result") is not None:
		print(json.dumps(result["vlm_result"], indent=2))
		if result.get("vlm_embedding") is not None:
			print()
			print(thin)
			print(" VLM Embedding")
			print(thin)
			print(json.dumps(result["vlm_embedding"], indent=2))
		if result.get("vlm_embedding_error"):
			print(f"  Embedding error: {result['vlm_embedding_error']}")
	elif result.get("vlm_error"):
		print(f"  Error: {result['vlm_error']}")
	else:
		print("  (no VLM result)")

	# ── Copy-Paste Prompt ────────────────────────────────────────────────
	print()
	print(sep)
	print(" COPY-PASTE PROMPT  (paste directly into OpenAI)")
	print(sep)

	lines: list[str] = []

	# System instruction
	lines.append("You are an expert NBA rules official. Use only the provided play description and rulebook excerpts.")
	lines.append("Do not infer unseen foot contacts between frames.")
	lines.append("If the evidence is insufficient to establish a violation, say the play is inconclusive rather than guessing.")
	lines.append("")

	# Question
	lines.append("Question: Was there enough visible evidence to rule this a travelling violation under the NBA rules?")
	lines.append("")

	# Decision standard
	lines.append("Decision standard:")
	lines.append("- A travelling violation should only be called if the visible evidence clearly establishes the gather point and an illegal foot movement after the gather.")
	lines.append("- If the gather point, pivot foot, or post-gather foot contacts are ambiguous, do not call a violation.")
	lines.append("- Distinguish between:")
	lines.append('  1. "No travel observed"')
	lines.append('  2. "Travel observed"')
	lines.append('  3. "Inconclusive / cannot determine from provided frames"')
	lines.append("")

	# Required analysis
	lines.append("Required analysis:")
	lines.append("1. Identify the most likely gather frame and explain why using Rule 4, Section III.")
	lines.append("2. Identify whether the player is progressing or stationary at gather.")
	lines.append("3. Identify the first and second step after gather under Rule 8, Section XIII(b), if visible.")
	lines.append("4. Identify whether a pivot foot is established, and whether it is lifted and returned.")
	lines.append("5. Identify whether the player jumps and lands with the ball.")
	lines.append("6. State whether the evidence clearly supports a travelling violation.")
	lines.append("")

	# Play description from VLM
	if result.get("vlm_result"):
		lines.append("--- Play Description ---")
		vlm = result["vlm_result"]
		if vlm.get("play_narration"):
			lines.append(vlm["play_narration"])
		if vlm.get("query_relevant_signals"):
			lines.append("")
			lines.append("Key signals observed:")
			for sig in vlm["query_relevant_signals"]:
				lines.append(f"- {sig}")
		if vlm.get("uncertainties"):
			lines.append("")
			lines.append("Uncertainties:")
			for unc in vlm["uncertainties"]:
				lines.append(f"- {unc}")
		lines.append("")

	# Rulebook excerpts
	if result.get("rag_chunks"):
		lines.append("--- Relevant Rulebook Excerpts ---")
		for i, chunk in enumerate(result["rag_chunks"], start=1):
			lines.append(f"[Excerpt {i} \u2014 {chunk.get('section_title', 'N/A')}]")
			lines.append(chunk["text"])
			lines.append("")

	# JSON output schema
	lines.append("Return JSON only:")
	lines.append(json.dumps({
		"question": "Was this player in blue uniform travelling?",
		"applicable_rules": [
			{"rule": "Rule 4, Section III(b)", "application": "How the gather rule applies to the visible facts."},
			{"rule": "Rule 8, Section XIII(b)", "application": "How the two-step and pivot-foot rules apply to the visible facts."},
		],
		"likely_gather_frame": "F01|F02|F03|F04|F05|F06|F07|unclear",
		"gather_confidence": "low|medium|high",
		"post_gather_step_count_visible": "0|1|2|more_than_2|unclear",
		"pivot_foot": "left|right|either|none|unclear",
		"travel_ruling": "travel|no_travel|inconclusive",
		"ruling_confidence": "low|medium|high",
		"reasoning": "Short explanation focused on gather, steps, and pivot foot.",
		"facts_required_for_travel_call": ["Specific missing or required fact."],
		"limitations": ["Specific ambiguity caused by sampled frames, occlusion, or missing video."],
	}, indent=2))

	print("\n".join(lines))
	print(sep)

	# ── Ruling ───────────────────────────────────────────────────────────
	if result.get("ruling"):
		r = result["ruling"]
		verdict_map = {
			"travel": "TRAVEL",
			"no_travel": "NO TRAVEL",
			"inconclusive": "INCONCLUSIVE",
		}
		confidence_map = {"low": "Low", "medium": "Medium", "high": "High"}
		verdict = verdict_map.get(r["travel_ruling"], r["travel_ruling"].upper())
		confidence = confidence_map.get(r["ruling_confidence"], r["ruling_confidence"].capitalize())

		print()
		print(sep)
		print(" RULING")
		print(sep)
		print(f"  Question  : {r['question']}")
		print(f"  Verdict   : {verdict}")
		print(f"  Confidence: {confidence}")
		print(f"  Gather    : Frame {r['likely_gather_frame']}  (confidence: {r['gather_confidence']})")
		print(f"  Steps visible after gather : {r['post_gather_step_count_visible']}")
		print(f"  Pivot foot: {r['pivot_foot']}")
		print()
		print(thin)
		print("  Applicable Rules")
		print(thin)
		for rule in r["applicable_rules"]:
			print(f"  {rule['rule']}")
			# word-wrap the application text at ~68 chars
			words = rule["application"].split()
			line_buf: list[str] = []
			for word in words:
				if sum(len(w) + 1 for w in line_buf) + len(word) > 68:
					print("    " + " ".join(line_buf))
					line_buf = [word]
				else:
					line_buf.append(word)
			if line_buf:
				print("    " + " ".join(line_buf))
			print()
		print(thin)
		print("  Reasoning")
		print(thin)
		words = r["reasoning"].split()
		line_buf = []
		for word in words:
			if sum(len(w) + 1 for w in line_buf) + len(word) > 68:
				print("  " + " ".join(line_buf))
				line_buf = [word]
			else:
				line_buf.append(word)
		if line_buf:
			print("  " + " ".join(line_buf))
		print()
		print(thin)
		print("  Facts Required for a Travel Call")
		print(thin)
		for fact in r["facts_required_for_travel_call"]:
			print(f"  - {fact}")
		print()
		print(thin)
		print("  Limitations")
		print(thin)
		for lim in r["limitations"]:
			print(f"  - {lim}")
		print()
		print(sep)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
