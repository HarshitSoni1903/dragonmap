#!/usr/bin/env python3
"""
End-to-end preparation pipeline for SeMRA.

Steps:
 1. Load config
 2. Create SQLite DB + tables + triggers
 3. Ingest concept_raw and priority_raw files (streaming, chunked)
 4. Apply Rule-A label updates everywhere
 5. Generate replacement pairs (old→new, new→old)
 6. Export cleaned SapBERT input files (sapbert_all + splits)
"""

import sys
import time
from pathlib import Path

from config_loader import PrepConfig, load_config
from db_utils import (
    create_tables,
    create_update_triggers,
    backfill_missing_labels,
    generate_replacement_pairs,
)
from ingestion import run_ingestion
from export_utils import (
    export_sapbert_all,
    export_train_val_test_split,
)


def banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def main():
    # ----------------------------------------
    # Load config
    # ----------------------------------------
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "prep_config.yaml"
    cfg: PrepConfig = load_config(cfg_path)

    banner(f"[CONFIG] Loaded configuration: {cfg_path}")
    print(cfg)

    # ----------------------------------------
    # Create DB + tables + triggers
    # ----------------------------------------
    banner("[DB] Creating tables and triggers…")
    create_tables(cfg, priority_union=cfg.priority_union)
    create_update_triggers(cfg)

    # ----------------------------------------
    # Ingestion (concept_raw + priority_raw)
    # ----------------------------------------
    banner("[INGESTION] Running ingestion…")
    start = time.time()
    run_ingestion(cfg)
    print(f"[INGESTION] Completed in {(time.time() - start):.2f}s")

    # ----------------------------------------
    # Initial backfill of missing labels
    # ----------------------------------------
    banner("[LABEL] Applying universal backfill (Rule-A)…")
    backfill_missing_labels(cfg)

    # ----------------------------------------
    # Generate replacement pairs
    # ----------------------------------------
    banner("[REPLACEMENTS] Generating replacement pairs…")
    generate_replacement_pairs(cfg)

    # ----------------------------------------
    # Export SapBERT training files
    # ----------------------------------------
    banner("[EXPORT] Exporting sapbert_all.csv …")
    export_sapbert_all(cfg)

    banner("[EXPORT] Generating train/val/test splits …")
    export_train_val_test_split(cfg)

    banner("[DONE] SeMRA preparation pipeline complete.")


if __name__ == "__main__":
    main()
