# DragonMAP

**Contributors:** Gabriel Nixon · Harshit Soni · Aditya Desai  

---

## 📄 Overview

This project proposes a **two-stage ontology mapping pipeline** that combines **DragonAI** (a Retrieval-Augmented Generation system) with **BERTMap** (a structure-aware ontology aligner).  
The integrated approach aims to improve **alignment quality, efficiency, and interpretability** by leveraging LLM-assisted completion for incomplete ontology data, followed by robust graph-based matching.

---

## 🎯 Objectives

- Achieve **high-precision and high-recall** mappings with statistical confidence.
- Use **RAG enrichment selectively**, focusing only where evidence is weak.
- Enable **1→N mappings** for varying ontology granularity, ensuring coherence and interpretability.
- Reduce manual curation through automated validation and reproducibility.

---

## ⚙️ Pipeline Overview

### 1. DragonAI (RAG)
- Addresses **missing or incomplete ontology data** (e.g., SNOMED, FMA).  
- Uses a **Vector Database** for efficient retrieval.  
- Applies a **LLM to impute missing synonyms, definitions, and parent links**.  

### 2. Candidate Generation & Pre-cleaning
- Combines **lexical top-k** and **embedding-based top-k** retrieval for diversity.  
- **Filters incompatible or disjoint pairs** based on semantic and structural constraints.  

### 3. Matching Score Computation
- Uses **BERTMap** to calculate source-target **mapping probability scores**.  

### 4. Thresholding
- Retains **top-N or probability-threshold** mappings.  
- Handles close probabilities (e.g., 0.51 vs 0.50) via contextual adjudication.  

### 5. Sanity Check / Validation
- Applies an **LLM for contextual verification**.  
- Compares results with **gold-standard mappings** and **BioRegistry cross-checks**.  


