# DragonAI–BERTMap + SeMRA (Approach 1)

This repository is a runnable scaffold for **Approach 1**:
using **SeMRA graphs as contextual knowledge** for DragonAI (RAG),
and **provenance-weighted fusion** with BERTMap scores to produce grounded mappings.

## Quickstart

```bash
cd dragonai-bertmap-semra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

## What it does (today)

- Loads a sample SeMRA TSV with **provenance and hop counts**.
- Converts SeMRA edges into **confidence priors** using a simple heuristic.
- Generates naive lexical/embedding candidates, **scores via mock BERTMap**, and
  **fuses** scores with SeMRA priors: `s* = alpha * s_b + (1 - alpha) * p_se`.
- Thresholds and evaluates @1 against a tiny gold file.

## Where to extend (tomorrow)

- Swap `mock_augment` with **real DragonAI RAG** (retriever + LLM).
- Replace `score_pairs` with **real BERTMap scoring**.
- Implement **ID normalization** for all ontology IRIs/CURIEs.
- Add **bioregistry** checks and full **OAEI** evaluation.

## Files

- `src/semra_ingest.py` — loads SeMRA and computes provenance-weighted priors.
- `src/dragonai_rag.py` — RAG interface with a mock augmentation method.
- `src/candidate_generation.py` — lexical + embedding candidate interleaving.
- `src/bertmap_integration.py` — placeholder for BERTMap scoring.
- `src/fusion.py` — score fusion (BERTMap × SeMRA prior).
- `src/thresholding.py` — top-k / tie-aware selection.
- `src/validation.py` — simple P@1/accuracy evaluation.
- `src/pipeline.py` — end-to-end glue for the above.

## Configuration

See `config.yaml` to tune:
- **provenance priors**, **hop decay**, **fusion alpha**,
- **retrieval top-k**, and **thresholds**.

## Notes

This scaffold is intentionally lightweight, with placeholder models and tiny sample data,
so you can plug in production components incrementally.
