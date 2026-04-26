# NBA Video Rules RAG — YouTube Clip Officiating Analysis

A multimodal RAG system that watches a short basketball clip, narrates it with a vision model, retrieves relevant NBA rulebook sections, and produces a structured officiating analysis with explicit rule citations and confidence levels.

## Pipeline

```
YouTube URL + start/end timestamps + question
              ↓
  1. Frame extraction     (yt-dlp + OpenCV, 5–10 keyframes)
              ↓
  2. Rulebook index       (auto-built from PDF if missing, FAISS + BAAI/bge-small-en-v1.5)
              ↓
  3. VLM narration        (any OpenAI-compatible vision model → structured JSON)
              ↓
  4. VLM output embedding (embed narration for downstream similarity use)
              ↓
  5. Query building       (VLM output + question → dense retrieval query)
              ↓
  6. FAISS retrieval      (top-5 rulebook chunks by semantic similarity)
              ↓
  7. Ruling prompt        (structured prompt assembled for manual or automated reasoning)
              ↓
  Answer + verdict + applicable rules + reasoning + confidence + limitations
```

## Current scope
- Focus: travelling calls (Rule 4 Section III — gather; Rule 8 Section XIII — two-step/pivot)
- Input: YouTube URL + start/end timestamps + question
- Vision: structured play narration extracted from 5–10 sampled frames
- RAG: semantic retrieval over the 2023–24 NBA Official Playing Rules
- Output: officiating-style ruling with `travel | no_travel | inconclusive` verdict, gather frame, step count, pivot foot, reasoning, and explicit facts-required list

## Models
| Stage | How to configure | Notes |
|---|---|---|
| VLM narration | `VLM_MODEL` env var | Any OpenAI-compatible vision model |
| Rulebook embeddings | `BAAI/bge-small-en-v1.5` (fixed) | Lightweight CPU sentence embeddings |
| Ruling / reasoning | manual copy-paste or `REASONER_MODEL` env var | Structured JSON prompt generated automatically |

Both VLM and reasoning stages target `https://api.openai.com/v1/chat/completions` by default. Override with `VLM_API_BASE` / `REASONER_API_BASE` env vars.

## Terminal output sections

Running `run_nba_query.py` produces clearly separated sections:

| Section | Content |
|---|---|
| **Header** | URL, video ID, interval, question, frame count, output paths |
| **Rulebook Embeddings** | Whether the index was built or already present, PDF path, chunk count |
| **VLM Prompt** | Full prompt sent to the vision model |
| **VLM Output** | Structured JSON narration returned by the VLM, plus embedding metadata |
| **Copy-Paste Prompt** | Complete ready-to-paste prompt for OpenAI with system instruction, play description, rulebook excerpts, decision standard, required analysis, and JSON output schema |
| **Ruling** *(demo mode)* | Formatted verdict, applicable rules, reasoning, required facts, and limitations |

## Project structure
```
src/nba_rules_rag/
  frame_extraction.py  # yt-dlp download + OpenCV keyframe sampling
  vlm_describer.py     # VLM API call, structured JSON narration
  query_builder.py     # VLM output → retrieval query string
  retriever.py         # FAISS semantic search over rulebook
  embeddings.py        # sentence-transformers embed + FAISS index build/save
  chunking.py          # rulebook PDF → rule sections → word-window chunks
  rulebook_loader.py   # pypdf text extraction
  youtube_utils.py     # URL parsing, timestamp parsing, interval validation
notebooks/
  nba_rules_rag_colab.ipynb  # Colab walkthrough
scripts/
  run_nba_query.py           # main CLI — frame extraction, VLM, RAG, ruling
data/
  vector_store/        # FAISS index + chunk metadata (built at runtime)
  processed/           # extracted frames and demo artifacts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Live mode only** (not needed for `--demo`): install `ffmpeg` as a system dependency — yt-dlp requires it to trim downloaded video segments.
>
> ```bash
> # macOS
> brew install ffmpeg
> # Ubuntu/Debian
> sudo apt install ffmpeg
> ```

The rulebook index is built automatically on first run if not already present.

## Usage

### Demo mode (no live API calls)

Uses a built-in clip and pre-saved VLM output. No `VLM_MODEL` or API key needed.

```bash
python scripts/run_nba_query.py --demo
```

### Live mode

```bash
export VLM_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

python scripts/run_nba_query.py \
  --youtube-url "https://www.youtube.com/watch?v=9fFWawcJXUw" \
  --start-time "0:21" \
  --end-time "0:29" \
  --question "Was the player in the blue uniform traveling?"
```

The **Copy-Paste Prompt** section at the end of the output contains the full structured prompt — paste it directly into ChatGPT or any OpenAI-compatible interface to get the JSON ruling.

### CLI reference

```
--youtube-url          YouTube URL to analyse
--start-time           Clip start (SS, MM:SS, or HH:MM:SS)
--end-time             Clip end
--question             Officiating question
--n-frames             Target keyframe count (clamped to 5–10, default 8)
--output-dir           Where to save frames and grid preview
--no-show              Skip interactive matplotlib display
--demo                 Use built-in clip + saved VLM output, no API calls
--rulebook-pdf-path    Path to NBA rulebook PDF (auto-used if index missing)
--vector-store-dir     FAISS index directory (default: data/vector_store)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLM_MODEL` | *(required for live mode)* | Vision model ID |
| `OPENAI_API_KEY` | *(required for live mode)* | API key |
| `VLM_API_BASE` | `https://api.openai.com/v1/chat/completions` | Override VLM endpoint |
| `REASONER_API_BASE` | `https://api.openai.com/v1/chat/completions` | Override reasoning endpoint |
| `EMBEDDING_DEVICE` | `cpu` | Set to `cuda` to use GPU for embeddings |

## Tests

```bash
pytest tests/ -m "not integration"
```

Integration tests (marked `@pytest.mark.integration`) require `yt-dlp` and network access.

## Progress

- [x] YouTube URL + timestamp validation and parsing
- [x] Frame extraction and keyframe sampling
- [x] Rulebook PDF ingestion, chunking, and FAISS index
- [x] Auto-build rulebook index on first run
- [x] VLM structured play narration (JSON schema with gather/step signals)
- [x] VLM output embedding saved alongside frames
- [x] Query builder combining VLM output + question for retrieval
- [x] Top-5 FAISS retrieval with section titles and similarity scores
- [x] Structured copy-paste ruling prompt (decision standard, JSON schema, rulebook excerpts)
- [x] Demo mode with pre-saved clip, VLM output, and ChatGPT ruling
- [x] Demo mode loads saved frames from data/processed/demo_frames
- [x] Clearly separated terminal output for each pipeline stage
- [x] Cleanup pass: removed deprecated scripts/modules and stale tests
- [ ] Automated reasoning step wired into CLI (currently copy-paste to OpenAI)
- [ ] Broader rule coverage beyond travelling (fouls, out-of-bounds, etc.)
- [ ] Colab notebook polish and end-to-end reproducibility improvements
- [ ] Demo UI (Gradio or Streamlit)

## License
MIT. See `LICENSE`.

