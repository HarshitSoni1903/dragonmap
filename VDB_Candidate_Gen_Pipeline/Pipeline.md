# DragonMap — Vector DB & Candidate Generation Pipeline

> **End-to-End Reproducible Setup — Local + HPC**
>
> HP + MP Ontologies → Enrichment → Embeddings → Qdrant Vector DB → Candidate Search

---

## Overview


- Convert OWL ontologies to CSV
- Enrich them using DragonAI (optional)
- Embed with SapBERT or fine-tuned SapBERT
- Store embeddings in Qdrant (vector DB)
- Query top-K candidates per label
- Export results (TSV, synonyms, definitions)

### Supported Modes

| Mode | Input | Model |
|------|-------|-------|
| 1 | Raw HP/MP OWL | SapBERT |
| 2 | Enriched HP/MP CSV | SapBERT |
| 3 | Raw HP/MP OWL | Fine-tuned SapBERT |
| 4 | Enriched HP/MP CSV | Fine-tuned SapBERT |

Everything runs with one pipeline file + simple commands.

---

## 1. Folder Structure (Required)

```
Pipeline/
│
├── Prod-Ready-pipeline/
│   ├── data/
│   │   ├── hp.owl
│   │   ├── mp.owl
│   │   ├── enriched_hp.csv        
│   │   ├── enriched_mp.csv        
│   │
│   ├── embeddings/                # auto-populated
│   │   ├── hp_sapbert.npy
│   │   ├── hp_sapbert_payloads.pkl
│   │   ├── mp_sapbert.npy
│   │   ├── mp_sapbert_payloads.pkl
│   │   ├── enriched_hp_sapbert.npy
│   │   ├── enriched_hp_sapbert_payloads.pkl
│   │   ├── enriched_mp_sapbert.npy
│   │   ├── enriched_mp_sapbert_payloads.pkl
│   │
│   ├── pipeline_v3.py            # embedding + caching + upload
│   ├── get_candidates_v5.py      # candidate generation + metadata
│   ├── candidate_generate.py     # helper
│   ├── debug_qdrant.py
│   ├── requirements.txt
│   ├── run_commands.md
│   └── README.md
```

---

## 2. Environment Setup

### Local (Mac/Linux)

Create Conda environment:

```bash
conda create -n dmpipeline python=3.10 -y
conda activate dmpipeline
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with the following contents:

```
qdrant-client==1.16.1
transformers
torch
pandas
numpy
rdflib
tqdm
scikit-learn
fasttext-wheel
```

---

### HPC (NYU Greene/Burst)

Load modules:

```bash
module load anaconda3
module load cuda/12.1   # ONLY if using GPU
```

Create HPC environment:

```bash
conda create -n dmpipeline python=3.10 -y
conda activate dmpipeline
pip install -r requirements.txt --no-cache-dir
```

GPU-enabled torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Start Qdrant Vector Database

> **Important:** You must run Qdrant before uploading embeddings or searching.

### Local (Docker)

```bash
docker run -p 6333:6333 \
    -v qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Open UI at: [http://localhost:6333](http://localhost:6333)

---

### HPC (IF Docker Allowed)

Use Qdrant local mode by modifying `pipeline_v3.py`:

**Change:**
```python
client = QdrantClient(url="http://localhost:6333")
```

**To:**
```python
client = QdrantClient(path="qdrant_hpc_storage")  # stores DB locally
```

This allows Qdrant to run fully embedded, no server needed.

---

## 4. Run the Embedding Pipeline

Use `pipeline_v3.py` — supports OWL and CSV automatically with caching for speed.

### Mode 1 — Raw OWL + SapBERT

```bash
python pipeline_v3.py --data data/mp.owl --model sapbert
python pipeline_v3.py --data data/hp.owl --model sapbert
```

### Mode 2 — Enriched CSV + SapBERT

```bash
python pipeline_v3.py --data data/enriched_mp.csv --model sapbert
python pipeline_v3.py --data data/enriched_hp.csv --model sapbert
```

### Mode 3 — Raw OWL + Fine-tuned SapBERT

```bash
python pipeline_v3.py --data data/mp.owl --model finetuned --model_path /path/to/model
python pipeline_v3.py --data data/hp.owl --model finetuned --model_path /path/to/model
```

### Mode 4 — Enriched CSV + Fine-tuned SapBERT

```bash
python pipeline_v3.py --data data/enriched_mp.csv --model finetuned --model_path /path/to/model
```

---

## 5. Verify Collections in Qdrant UI

You should see the following collections:

- `hp_sapbert`
- `mp_sapbert`
- `enriched_hp_sapbert`
- `enriched_mp_sapbert`



## 6. Candidate Generation

Use `get_candidates_v5.py` which supports:

- Synonyms
- Definitions
- TSV export
- Metadata lookup

### Example (Raw)

```bash
python get_candidates_v5.py \
    --label "abnormal lung morphology" \
    --target_collection mp_sapbert \
    --model sapbert \
    --metadata data/mp.csv \
    --topk 20
```

### Example (Enriched)

```bash
python get_candidates_v5.py \
    --label "abnormal lung morphology" \
    --target_collection enriched_mp_sapbert \
    --model sapbert \
    --metadata data/enriched_mp.csv \
    --topk 20
```

TSV will be saved as: `candidates_<collection>_<label>.tsv`

---

## 7. End-to-End Flow Explanation

```
+-------------------------------------------------------------------------+
|                         DRAGONMAP PIPELINE                              |
+-------------------------------------------------------------------------+
|                                                                         |
|   +--------------+     +--------------+     +--------------+            |
|   |  HP/MP OWL   |---->|   Enrich     |---->|    Embed     |            |
|   |  Ontologies  |     |  (DragonAI)  |     |  (SapBERT)   |            |
|   +--------------+     +--------------+     +--------------+            |
|                              |                     |                    |
|                              v                     v                    |
|                        +--------------+     +--------------+            |
|                        |   CSV with   |     |   768-dim    |            |
|                        |  Synonyms &  |     |   Vectors    |            |
|                        | Definitions  |     |              |            |
|                        +--------------+     +--------------+            |
|                                                    |                    |
|                                                    v                    |
|                                             +--------------+            |
|                                             |    Qdrant    |            |
|                                             |  Vector DB   |            |
|                                             +--------------+            |
|                                                    |                    |
|                                                    v                    |
|   +--------------+     +--------------+     +--------------+            |
|   |  TSV Export  |<----|   Top-K      |<----|    Query     |            |
|   |  Candidates  |     |   Search     |     |   Embedding  |            |
|   +--------------+     +--------------+     +--------------+            |
|                                                                         |
+-------------------------------------------------------------------------+
```
