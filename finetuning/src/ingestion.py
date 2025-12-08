# ingestion.py
"""
Ingestion stage for SeMRA preparation pipeline.

Responsibilities:
 - Stream all concept_raw and priority_raw TSV files
 - Insert rows into DB using db_utils
 - Apply Rule A label logic through update_label_everywhere()
 - Chunked ingestion for memory safety
 - Clean, deterministic, single-run behavior
"""

import csv
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any

from config_loader import PrepConfig
from db_utils import (
    get_connection,
    insert_concept_raw,
    insert_priority_raw,
    update_label_everywhere
)


# ============================================================
# Helper: stream rows from gzipped TSV
# ============================================================

def _stream_tsv(path: Path, chunk_size: int):
    """Yield chunks of rows from a .tsv.gz file as lists of dicts."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


# ============================================================
# Ingest Concepts
# ============================================================

def ingest_concepts(cfg: PrepConfig, headers: Dict[str, Any]) -> None:
    """
    Ingest all concept tables and update concept labels using Rule A.
    """

    conn = get_connection(cfg)
    concept_pattern = "{landscape}_concept*.tsv.gz"
    concept_table = cfg.tables["concept_raw"]
    priority_table = cfg.tables["priority_raw"]
    epairs_table = cfg.tables["enriched_pairs"]

    print("=== Ingesting Concepts ===")

    for landscape in cfg.landscapes:
        files = sorted(cfg.data_dir.glob(concept_pattern.format(landscape=landscape)))
        if not files:
            print(f"[WARN] No concept files for landscape: {landscape}")
            continue

        for path in files:
            print(f"[INFO] Concept file: {path}")

            for chunk in _stream_tsv(path, cfg.chunk_size):
                for row in chunk:
                    curie = row.get("curie:ID")
                    label = row.get("name")

                    # Normalize
                    if cfg.ingestion["normalize_empty_strings"]:
                        if label is not None and label.strip() == "":
                            label = None

                    if curie is None:
                        continue

                    # Insert raw row (first appearance only)
                    insert_concept_raw(conn, curie, label)

                    # Apply rule A label propagation
                    update_label_everywhere(
                        conn,
                        epairs_table,
                        concept_table,
                        priority_table,
                        curie,
                        label
                    )

                conn.commit()

    conn.close()


# ============================================================
# Ingest Priority
# ============================================================

def ingest_priority(cfg: PrepConfig, headers: Dict[str, Any]) -> None:
    """
    Ingest all priority mappings into priority_raw.
    """

    conn = get_connection(cfg)
    priority_pattern = "{landscape}_priority*.tsv.gz"
    table = cfg.tables["priority_raw"]
    union_cols = headers["priority_union"]

    print("=== Ingesting Priority Mappings ===")

    for landscape in cfg.landscapes:
        files = sorted(cfg.data_dir.glob(priority_pattern.format(landscape=landscape)))
        if not files:
            print(f"[WARN] No priority files for landscape: {landscape}")
            continue

        for path in files:
            print(f"[INFO] Priority file: {path}")

            for chunk in _stream_tsv(path, cfg.chunk_size):

                # Optional chunk-level dedupe
                if cfg.ingestion["drop_duplicates"]:
                    seen = set()
                    dedup_chunk = []
                    for row in chunk:
                        key = tuple(row.items())
                        if key not in seen:
                            seen.add(key)
                            dedup_chunk.append(row)
                    chunk = dedup_chunk

                # Process chunk
                for row in chunk:
                    # normalize empty strings
                    if cfg.ingestion["normalize_empty_strings"]:
                        for k, v in row.items():
                            if v is not None and v.strip() == "":
                                row[k] = None

                    # Build rowdict using union columns
                    rowdict = {}
                    for col in union_cols:
                        rowdict[col] = row.get(col)  # missing columns → None

                    # Insert priority row
                    insert_priority_raw(conn, table, landscape, rowdict)

                conn.commit()

    conn.close()


# ============================================================
# MAIN ENTRY
# ============================================================

def run_ingestion(cfg: PrepConfig, headers: Dict[str, Any]) -> None:
    """Entry point called from prep.py"""
    ingest_concepts(cfg, headers)
    ingest_priority(cfg, headers)


# ============================================================
# BASIC UNIT TEST
# ============================================================

if __name__ == "__main__":
    from config_loader import load_config
    from header_utils import load_headers

    cfg = load_config()
    headers = load_headers(cfg)

    print("[TEST] Running ingestion...")
    run_ingestion(cfg, headers)

    print("[TEST] Ingestion complete.")
