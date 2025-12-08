# prep.py
"""
Top-level orchestrator for SeMRA dataset preparation.

Executes the pipeline in the following order:
1. Load configuration
2. Setup logging (redirect stdout/stderr to log file)
3. Load headers (concept/priority)
4. Create database schema
5. Create triggers
6. Ingest concept data
7. Ingest priority data
8. Build indexes
9. Initial backfill pass
10. SemRA enrichment
11. OBO enrichment
12. Export enriched_pairs
"""

from __future__ import annotations

from config_loader import load_config
from logger_setup import setup_logging
from header_utils import load_headers

# NEW: import trigger creation
from db_utils import (
    create_tables,
    create_ingestion_indexes,
    create_auto_enrich_trigger,
    initial_backfill_labels,
    count_enriched_pairs,
)

from ingestion import ingest_concepts, ingest_priority
from semra_generator import run_semra_enrichment
from obo_enrichment import run_obo_enrichment
from export_utils import export_sapbert_pairs


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(cfg):
    print("\n=======================================")
    print(" SeMRA Dataset Prep Started")
    print("=======================================\n")

    # --------------------------------------------------------
    # 1. HEADERS
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_header_loading", False):
        print("[PIPELINE] Skipping header loading per config.")
        headers = None
    else:
        print("[PIPELINE] Loading headers...")
        headers = load_headers(cfg)

    # --------------------------------------------------------
    # 2. DATABASE CREATION
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_db_creation", False):
        print("[PIPELINE] Skipping DB creation per config.")
    else:
        print("[PIPELINE] Creating database schema...")
        create_tables(cfg, headers["priority_union"])

        # >>> NEW: TRIGGER CREATION <<<
        print("[PIPELINE] Creating auto-enrich trigger...")
        create_auto_enrich_trigger(cfg)

    # --------------------------------------------------------
    # 3. INGEST CONCEPTS
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_concept_ingestion", False):
        print("[PIPELINE] Skipping concept ingestion per config.")
    else:
        print("[PIPELINE] Ingesting concepts...")
        ingest_concepts(cfg, headers, first_run=True)
        count_enriched_pairs(cfg)

    # --------------------------------------------------------
    # 4. INGEST PRIORITY MAPPINGS
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_priority_ingestion", False):
        print("[PIPELINE] Skipping priority ingestion per config.")
    else:
        print("[PIPELINE] Ingesting priority mappings...")
        ingest_priority(cfg, headers)
        count_enriched_pairs(cfg)

    # --------------------------------------------------------
    # 4.a. CREATE INDEXES
    # --------------------------------------------------------
    if (
        not cfg.pipeline.get("skip_concept_ingestion", False)
        and not cfg.pipeline.get("skip_priority_ingestion", False)
    ):
        print("[PIPELINE] Initial ingestion completed — creating indexes...")
        create_ingestion_indexes(cfg)

    # --------------------------------------------------------
    # 4.b INITIAL BACKFILL PASS
    # --------------------------------------------------------
    if (
        not cfg.pipeline.get("skip_concept_ingestion", False)
        and not cfg.pipeline.get("skip_priority_ingestion", False)
    ):
        print("[PIPELINE] Running initial label backfill...")
        initial_backfill_labels(
            cfg,
            cfg.tables["concept_raw"],
            cfg.tables["priority_raw"],
            cfg.tables["enriched_pairs"],
        )
        count_enriched_pairs(cfg)

    # --------------------------------------------------------
    # 5. SEMRA ENRICHMENT
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_semra_enrichment", False):
        print("[PIPELINE] Skipping SemRA enrichment per config.")
    else:
        print("[PIPELINE] Running SemRA enrichment...")
        run_semra_enrichment(cfg, headers)
        count_enriched_pairs(cfg)

    # --------------------------------------------------------
    # 6. OBO ENRICHMENT
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_obo_enrichment", False):
        print("[PIPELINE] Skipping OBO enrichment per config.")
    else:
        print("[PIPELINE] Running OBO enrichment...")
        run_obo_enrichment(cfg)
        count_enriched_pairs(cfg)

    # --------------------------------------------------------
    # 7. EXPORT
    # --------------------------------------------------------
    if cfg.pipeline.get("skip_export_pairs", False):
        print("[PIPELINE] Skipping export of enriched_pairs per config.")
    else:
        print("[PIPELINE] Exporting SapBERT-ready CSVs...")
        export_sapbert_pairs(cfg)

    print("\n=======================================")
    print(" SeMRA Dataset Prep Completed")
    print("=======================================\n")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    cfg = load_config()
    redirector = setup_logging(cfg)

    try:
        run_pipeline(cfg)
    finally:
        redirector.deactivate()
