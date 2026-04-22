# NBA Rules RAG

A Colab-friendly multimodal RAG prototype that analyzes a short basketball clip and retrieves relevant NBA rulebook sections to produce a grounded rules interpretation.

## MVP
- Input: YouTube link + timestamps
- Extract key frames
- Use a VLM to describe the play
- Retrieve relevant NBA rulebook sections
- Return a grounded answer with citations

## Project structure
- `src/nba_rules_rag/` - main Python package
- `notebooks/` - Colab notebook
- `scripts/` - helper scripts
- `data/` - local artifacts
- `tests/` - tests
