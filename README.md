# Lloyd's Property MVP – Starter Scaffold

This repo is a Cursor-friendly, Streamlit-based starter for a Lloyd’s-ready Property risk assessment MVP.
It includes:
- Streamlit UI
- LLM-driven extraction using Structured Outputs (placeholders)
- Lloyd’s-aligned JSON Schemas (Submission, SOV, Loss Run)
- COPE scoring + LLM Nuanced Concerns layer (placeholders)
- Indicative pricing (base × modifiers)
- Email ingestion with attachments

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env  # then edit your OpenAI key
streamlit run app/app.py
```

## Layout

- `app/` – Streamlit UI (tabs for Upload, Submission, SOV, Loss, Risk & Pricing)
- `core/` – core modules (extraction, parsing, risk, pricing, schemas, storage, utils)
- `backend/` – optional FastAPI endpoints if you want to split concerns later
- `tests/` – test stubs
- `data/samples/` – sample files for local testing
- `configs/` – configuration defaults

## Next steps
1. Implement OpenAI Structured Outputs in `core/extraction/structured_outputs.py`.
2. Wire parsers in `core/parsing/*` to provide normalized content to the extractor.
3. Implement COPE scoring in `core/risk/cope_rules.py` and nuanced concerns in `core/risk/nuanced.py`.
4. Fill base rates & modifiers in `core/pricing/engine.py`.
5. Hook everything up in `app/app.py`.
