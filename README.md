# NBA Rules RAG

A Colab-friendly multimodal RAG prototype that analyzes a short basketball clip and retrieves relevant NBA rulebook sections to produce a grounded rules interpretation.

## MVP scope (this week)
- Focus: traveling calls only (v1)
- Input: YouTube URL + start/end timestamps + optional question
- Process: sample 5-10 frames across the selected interval
- Vision: generate structured play summary from sampled frames
- RAG: retrieve official NBA rulebook chunks (priority: Rule 4 and Rule 10)
- Output: likely ruling, explanation, citations, and uncertainty

## Colab-first constraints
- This project is designed to run in Google Colab.
- No local LLM/VLM serving and no fine-tuning.
- Use hosted inference APIs from Colab runtime.

## Model choice (free/open)
- VLM: `Qwen/Qwen2.5-VL-3B-Instruct` (open-weight) via hosted inference API (for example, Hugging Face Inference).
- Reasoning model: hosted text model API from Colab (no local model execution).

## Project tracker
Status legend: [x] implemented, [ ] in progress/next

- [x] Repository scaffold and package layout under `src/`, `scripts/`, `notebooks/`, `tests/`
- [x] Minimal pipeline stub (`run_pipeline`) wired in package
- [ ] YouTube URL + timestamp validation/parsing
- [ ] Frame extraction and 5-10 keyframe sampling in selected interval
- [ ] VLM frame-to-JSON play description step
- [ ] Query builder for traveling-focused rule retrieval
- [ ] Rulebook ingestion and subsection chunking with metadata
- [ ] Vector store index (FAISS/Chroma) over official NBA rulebook text
- [ ] Top-k retrieval with source URLs and rule metadata
- [ ] Grounded reasoning output with citations + uncertainty
- [ ] Demo UI (Colab notebook first) showing input, keyframes, summary, ruling
- [ ] Tests for chunking, query builder, and pipeline smoke path

## Project structure
- `src/nba_rules_rag/` - main Python package
- `notebooks/` - Colab notebook
- `scripts/` - helper scripts
- `data/` - local artifacts
- `tests/` - tests

## Setup and run

### Option A (recommended): Google Colab
1. Open `notebooks/nba_rules_rag_colab.ipynb` in Colab.
2. Install dependencies in the first cell:
	```bash
	!pip install -r requirements.txt
	!pip install -e .
	```
3. Run all notebook cells in order.

### Option B: Local smoke check (development)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the current pipeline stub:
```bash
python -c "from nba_rules_rag.pipeline import run_pipeline; print(run_pipeline('https://www.youtube.com/watch?v=abc123','00:41','00:45','Was this a travel?'))"
```

Expected result: a JSON-like dictionary with input fields and `status: pipeline stub created`.

## Current run status
- `scripts/build_rulebook_index.py` and `scripts/run_demo.py` are placeholders and will be wired as the next implementation steps.

## License
MIT. See `LICENSE`.
