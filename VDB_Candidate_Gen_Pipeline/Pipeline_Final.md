# DragonMap — Vector DB & Candidate Generation Pipeline

> **End-to-End Reproducible Setup — Local + HPC**  
> HP + MP Ontologies → Enrichment → Embeddings → Qdrant Vector DB → Candidate Search → CSV exports

---

## Overview

- Convert OWL ontologies to CSV
- Enrich them using DragonAI (optional)
- Embed with SapBERT or fine-tuned SapBERT
- Store embeddings in Qdrant (vector DB)
- Query top-K candidates per label
- Export results as CSV (with synonyms + definitions)
- Run a fixed “hard-case” benchmark suite and save all candidate sets

### Supported Modes

| Mode | Input            | Model                      |
|------|------------------|----------------------------|
| 1    | Raw HP/MP OWL    | SapBERT                    |
| 2    | Enriched HP/MP   | SapBERT                    |
| 3    | Raw HP/MP OWL    | Fine-tuned SapBERT (seMRA) |
| 4    | Enriched HP/MP   | Fine-tuned SapBERT (seMRA) |

Everything runs with one embedding pipeline + simple candidate scripts.

---

## 1. Folder Structure (Required)

```bash
Pipeline/
│
├── Prod-Ready-pipeline/
│   ├── data/
│   │   ├── hp.owl
│   │   ├── mp.owl
│   │   ├── enriched_hp.csv
│   │   ├── enriched_mp.csv
│   │   └── mp-hp-equivalent-2019-02-13.csv
│   │
│   ├── embeddings/
│   ├── pipeline_v3.py
│   ├── get_candidates_v5.py
│   ├── get_candidates_v5_csv.py
│   ├── run_hardcases.py
│   ├── debug_qdrant.py
│   └── Pipeline.md
│
├── qdrant_storage/
└── hardcase_outputs/
```

---

## 2. Environment Setup

### Local (Mac/Linux)

```bash
conda create -n dmpipeline python=3.10 -y
conda activate dmpipeline
pip install -r requirements.txt
```

### HPC (NYU Greene)

```bash
module load anaconda3
conda create -n dmpipeline python=3.10 -y
conda activate dmpipeline
pip install -r requirements.txt --no-cache-dir
```

---

## 3. Start Qdrant

### Local (Docker)

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### HPC (Embedded mode)

```python
client = QdrantClient(path="qdrant_hpc_storage")
```

---

## 4. Embedding Pipeline (pipeline_v3.py)

### Raw OWL + SapBERT

```bash
python pipeline_v3.py --data data/mp.owl --model sapbert
python pipeline_v3.py --data data/hp.owl --model sapbert
```

### Enriched CSV + SapBERT

```bash
python pipeline_v3.py --data data/enriched_mp.csv --model sapbert
python pipeline_v3.py --data data/enriched_hp.csv --model sapbert
```

### Fine-Tuned SapBERT (seMRA)

```bash
python pipeline_v3.py --data data/mp.owl --model seMRA_ms_sapbert_v1
python pipeline_v3.py --data data/enriched_mp.csv --model seMRA_ms_sapbert_v1
```

---

## 5. Candidate Retrieval

### Interactive (print only)

```bash
python get_candidates_v5.py    --label "Hypoalbuminemia"    --target_collection enriched_mp_sapbert    --model sapbert    --metadata data/enriched_mp.csv    --topk 20
```

### Save CSV

```bash
python get_candidates_v5.py    --label "Hypoalbuminemia"    --target_collection enriched_mp_sapbert    --model sapbert    --metadata data/enriched_mp.csv    --topk 20    --out_csv outputs/Hypoalbuminemia__enriched_mp_sapbert__top20.csv
```

---

## 6. Batch Hard-Case Runner

```bash
python run_hardcases.py    --collection enriched_mp_sapbert    --metadata data/enriched_mp.csv    --topk 25    --out_dir hardcase_outputs
```

Creates files like:

```
Hypoalbuminemia__enriched_mp_sapbert__top25.csv
Micrognathia__enriched_mp_sapbert__top25.csv
...
```

---

## 7. End-to-End Flow

```
OWL/CSV → pipeline_v3.py → Qdrant Vector DB
       → get_candidates_v5.py (manual)
       → get_candidates_v5_csv.py (CSV)
       → run_hardcases.py (batch evaluation)
```

A complete reproducible ontology-matching pipeline for HP ↔ MP with SapBERT + fine‑tuned SapBERT support.
