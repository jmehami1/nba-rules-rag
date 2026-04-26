# NBA Video Rules RAG — YouTube Clip Officiating Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmehami1/nba-rules-rag/blob/main/notebooks/nba_rules_rag_colab.ipynb)

A multimodal RAG system that watches a short basketball clip, narrates it with a vision model, retrieves relevant NBA rulebook sections, and produces a structured officiating analysis with explicit rule citations and confidence levels.

## Try it in Colab

The fastest way to run the full pipeline is the interactive Colab notebook — no local setup required:

**[Open `nba_rules_rag_colab.ipynb` in Colab](https://colab.research.google.com/github/jmehami1/nba-rules-rag/blob/main/notebooks/nba_rules_rag_colab.ipynb)**

The notebook walks through every stage of the pipeline with clearly labelled sections, inline outputs, and a final ruling display. Run it top-to-bottom. Demo mode (`DEMO_MODE = True`) requires no API key.

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
  7. Automated reasoning  (OpenAI-compatible endpoint → structured JSON ruling)
              ↓
  Verdict + applicable rules + reasoning + confidence + limitations
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
| Ruling / reasoning | `REASONER_MODEL` env var | Automated via OpenAI-compatible endpoint; copy-paste prompt also printed as fallback |

Both VLM and reasoning stages target `https://api.openai.com/v1/chat/completions` by default. Override with `VLM_API_BASE` / `REASONER_API_BASE` env vars.

## Terminal output sections

Running `run_nba_query.py` produces clearly separated sections:

| Section | Content |
|---|---|
| **Header** | URL, video ID, interval, question, frame count, output paths |
| **Rulebook Embeddings** | Whether the index was built or already present, PDF path, chunk count |
| **VLM Prompt** | Full prompt sent to the vision model |
| **VLM Output** | Raw VLM model response first, then parsed structured JSON narration, plus embedding metadata |
| **Copy-Paste Prompt** | Complete ready-to-paste prompt for manual review in ChatGPT or any OpenAI-compatible interface |
| **Ruling** | Formatted verdict, applicable rules, reasoning, required facts, and limitations (automated in live mode; pre-computed in demo mode) |
| **Status Report** | Final per-stage result summary with icons and `successful` or `failure` status labels |

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
  nba_rules_rag_colab.ipynb  # end-to-end Colab walkthrough (demo + live modes)
scripts/
  run_nba_query.py           # main CLI — frame extraction, VLM, RAG, ruling
data/
  vector_store/        # FAISS index + chunk metadata (built at runtime)
  processed/           # extracted frames and demo artifacts
    demo_frames/       # pre-saved frames for --demo mode (no API calls needed)
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

Uses a built-in clip with saved demo frames and a local mock VLM output. No API key or model config needed.

```bash
python scripts/run_nba_query.py --demo
```

### Live mode

On the first live run, the CLI will interactively prompt for any missing credentials and model choices. You will not need to set environment variables manually unless you prefer to:

```bash
python scripts/run_nba_query.py \
  --youtube-url "https://www.youtube.com/watch?v=9fFWawcJXUw" \
  --start-time "0:21" \
  --end-time "0:29" \
  --question "Was the player in the blue uniform traveling?"
```

The CLI will prompt for:
- **`OPENAI_API_KEY`** — entered via a hidden prompt (not echoed to terminal)
- **`VLM_MODEL`** — selected from a numbered menu of known OpenAI models or a custom ID
- **`REASONER_MODEL`** — same selection menu

Before making any live VLM or reasoning API call, the CLI also shows a one-time warning that calls may charge your account and asks for confirmation:

```text
Warning: this run will call external LLM/VLM APIs using your API key and may charge your account.
Continue with paid API calls? [y/N]:
```

If you answer `N` (or press Enter), paid API calls are skipped and the run still completes with failure status entries for those skipped stages in the final status report.

Credentials are saved to a local `.env` file with permissions `600` (owner read/write only). The `.env` file is excluded from git via `.gitignore` and is **never committed to the repository**.

On subsequent runs the saved `.env` is loaded automatically and no prompts appear.

To set credentials manually instead:

```bash
export OPENAI_API_KEY=<your-key>
export VLM_MODEL=gpt-4o
export REASONER_MODEL=gpt-4o
```

Once configured, the pipeline runs fully automated: frames are extracted, the VLM narrates the play, the top-5 rulebook chunks are retrieved, and a structured **RULING** section is printed in the terminal.

The **Copy-Paste Prompt** section is also always printed as a fallback for manual review in ChatGPT or any OpenAI-compatible interface.

### Output directory behavior

- **Default output root**: `data/processed`
- **Demo mode** (`--demo`): uses `data/processed/demo_frames`
- **Live mode** (non-demo): writes run artifacts into a timestamped subdirectory under the output root, e.g. `data/processed/20260426_153045`

This keeps live runs isolated from each other while preserving stable demo assets.

### CLI reference

```
--youtube-url          YouTube URL to analyse
--start-time           Clip start (SS, MM:SS, or HH:MM:SS)
--end-time             Clip end
--question             Officiating question
--n-frames             Target keyframe count (clamped to 5–10, default 8)
--output-dir           Where to save frames and grid preview
--no-show              Skip interactive matplotlib display
--demo                 Use built-in clip + local mock VLM output, no API calls
--rulebook-pdf-path    Path to NBA rulebook PDF (auto-used if index missing)
--vector-store-dir     FAISS index directory (default: data/vector_store)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(prompted on first live run)* | API key — saved to `.env` (mode 600), never printed |
| `VLM_MODEL` | *(prompted on first live run)* | Vision model ID |
| `REASONER_MODEL` | *(prompted on first live run)* | Reasoning model ID; falls back to `VLM_MODEL` if unset |
| `VLM_API_BASE` | `https://api.openai.com/v1/chat/completions` | Override VLM endpoint |
| `REASONER_API_BASE` | `https://api.openai.com/v1/chat/completions` | Override reasoning endpoint |
| `VLM_REASONING_EFFORT` | `none` | Reasoning effort value passed to VLM API payload |
| `VLM_MAX_COMPLETION_TOKENS` | `2048` | Completion token cap for VLM API payload |
| `HF_TOKEN` | *(unset)* | Optional fallback token for non-OpenAI-compatible endpoints |
| `EMBEDDING_DEVICE` | `cpu` | Set to `cuda` to use GPU for embeddings |

## Credential security

- **`.env` is in `.gitignore`** and is never tracked or committed.
- The API key is entered via a hidden prompt (`getpass`) — it is never echoed to the terminal or printed anywhere.
- The `.env` file is written with `chmod 600` (owner read/write only).
- No credentials appear in any source file, notebook output, or test fixture.
- To rotate or remove saved credentials, delete or edit `.env` at the repo root.

## Tests

```bash
pytest tests/ -m "not integration"
```

Integration tests (marked `@pytest.mark.integration`) require `yt-dlp` and network access.

Real paid API tests are marked `@pytest.mark.api` and are skipped by default. Run them only with explicit opt-in:

```bash
pytest --run-api-tests tests/test_run_nba_query.py::test_api_calls_work -s -v
```

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
- [x] Demo mode with pre-saved clip and VLM output (no API calls)
- [x] Demo mode loads saved frames from `data/processed/demo_frames/`
- [x] Clearly separated terminal output for each pipeline stage
- [x] Automated reasoning via OpenAI-compatible endpoint (live mode)
- [x] Secure credential bootstrap: interactive prompts, `getpass`, `.env` + `chmod 600`
- [x] Model selection menu (numbered list of known OpenAI models + custom entry)
- [x] Cleanup pass: removed deprecated scripts/modules and stale tests
- [x] Colab notebook — full end-to-end walkthrough with demo and live modes, inline outputs, and structured final ruling display
- [ ] Demo UI (Gradio or Streamlit)

## Future work
- [ ] Broader rule coverage beyond travelling (fouls, out-of-bounds, etc.)

## License
MIT. See `LICENSE`.

