# semra_generator.py
"""
SemRA-based enrichment stage for SeMRA preparation pipeline.

Responsibilities:
 - Stream unenriched priority_raw rows per landscape from SQLite
 - Convert to SSSOM-style DataFrames for SemRA
 - Call SemRA to complete subject/object labels
 - Apply enrichment rules:
      * If DB label is missing and SemRA returns a label → fill it
      * If DB label exists and SemRA returns a different label → record conflict pairs
      * Never overwrite existing labels (skip_existing_labels = True)
 - Use update_label_everywhere() to propagate accepted labels
 - Rely on DB triggers to auto-mark rows enriched when both labels are present
 - Print clear per-chunk enrichment statistics
"""

from __future__ import annotations

import sys
import csv
import gzip
from typing import Dict, Any, List, Optional

import pandas as pd

from config_loader import PrepConfig
from db_utils import get_connection, update_label_everywhere

# SemRA imports – aligned with your template usage
from semra.io import from_sssom_df, get_sssom_df


# ============================================================
# Helpers
# ============================================================

def _count_missing(df: pd.DataFrame) -> Dict[str, int]:
    """Count missing subject/object/both labels in a DataFrame."""
    sub_missing = df["subject_label"].isna().sum() if "subject_label" in df.columns else len(df)
    obj_missing = df["object_label"].isna().sum() if "object_label" in df.columns else len(df)
    both_missing = ((df.get("subject_label").isna()) & (df.get("object_label").isna())).sum() \
        if "subject_label" in df.columns and "object_label" in df.columns else 0

    return {
        "sub_missing": int(sub_missing),
        "obj_missing": int(obj_missing),
        "both_missing": int(both_missing),
    }


def _ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure subject_label and object_label columns exist."""
    if "subject_label" not in df.columns:
        df["subject_label"] = None
    if "object_label" not in df.columns:
        df["object_label"] = None
    return df


def _get_sssom_cols(priority_headers: Dict[str, List[str]], landscape: str) -> List[str]:
    """
    Decide which columns to pass to SemRA for a given landscape.

    We start from that landscape's priority headers, but make sure
    subject_id, object_id, subject_label, object_label are included.
    """
    cols = list(priority_headers.get(landscape, []))

    for required in ["subject_id", "object_id", "subject_label", "object_label"]:
        if required not in cols:
            cols.append(required)

    return cols


def _load_priority_chunk(
    conn,
    cfg: PrepConfig,
    landscape: str,
    sssom_cols: List[str],
) -> pd.DataFrame:
    """
    Load one chunk of unenriched priority rows for a given landscape.

    We only select rows where added_to_enrich = 0. No OFFSET is used; rows
    disappear from the unenriched pool as they get enriched.
    """
    table = cfg.tables["priority_raw"]
    col_sql = ", ".join(f'"{c}"' for c in sssom_cols)

    query = f"""
        SELECT {col_sql}
        FROM {table}
        WHERE landscape = ?
          AND added_to_enrich = 0
        ORDER BY rowid
        LIMIT {cfg.chunk_size};
    """

    return pd.read_sql_query(query, conn, params=(landscape,))


def _insert_conflict_pairs(cur, epairs_table: str, curie_id: str,
                           existing_label: str, semra_label: str) -> None:
    """
    When DB has an existing label and SemRA returns a different label,
    add both conflict pairs:

        (id, existing) → (id, semra)
        (id, semra)   → (id, existing)
    """
    if existing_label is None or semra_label is None:
        return
    if existing_label == semra_label:
        return

    cur.execute(
        f"""
        INSERT INTO {epairs_table}
        (subject_id, subject_label, object_id, object_label)
        VALUES (?, ?, ?, ?)
        """,
        (curie_id, existing_label, curie_id, semra_label),
    )
    cur.execute(
        f"""
        INSERT INTO {epairs_table}
        (subject_id, subject_label, object_id, object_label)
        VALUES (?, ?, ?, ?)
        """,
        (curie_id, semra_label, curie_id, existing_label),
    )


# ============================================================
# Main enrichment logic
# ============================================================

def enrich_landscape(cfg: PrepConfig, headers: Dict[str, Any], landscape: str) -> None:
    """
    Run SemRA-based enrichment for a single landscape.

    Steps:
      - Repeatedly fetch chunks of unenriched rows (added_to_enrich = 0)
      - Run SemRA to fill missing labels
      - Apply label update rules and record conflicts
      - Let triggers mark rows enriched when both labels are present
    """
    if cfg.semra_options["enable_label_completion"] is False:
        print(f"[{landscape}] SemRA label completion disabled in config; skipping.")
        return

    conn = get_connection(cfg)
    cur = conn.cursor()

    priority_headers = headers.get("priority_headers", {})
    sssom_cols = _get_sssom_cols(priority_headers, landscape)

    priority_table = cfg.tables["priority_raw"]
    concept_table = cfg.tables["concept_raw"]
    epairs_table = cfg.tables["enriched_pairs"]

    chunk_idx = 0
    total_filled_sub = 0
    total_filled_obj = 0
    total_conflicts = 0

    print(f"\n=== Enrichment: {landscape} ===")

    while True:
        df_chunk = _load_priority_chunk(conn, cfg, landscape, sssom_cols)

        if df_chunk.empty:
            break

        chunk_idx += 1
        print(f"\n[{landscape}] Chunk {chunk_idx} → {len(df_chunk)} rows")

        # Normalize empty strings to None
        df_chunk = df_chunk.replace({"": None})

        # BEFORE stats
        before = _count_missing(_ensure_label_columns(df_chunk))
        print(f"[{landscape}] BEFORE  → "
              f"sub_missing={before['sub_missing']}, "
              f"obj_missing={before['obj_missing']}, "
              f"both_missing={before['both_missing']}")

        # Build SemRA object and enrich labels
        try:
            semra_obj = from_sssom_df(df_chunk)
            df_completed = get_sssom_df(semra_obj, add_labels=True)
        except Exception as e:
            if cfg.semra_options.get("allow_external_resolution_errors", True):
                print(f"[{landscape}] [WARN] SemRA failed on chunk {chunk_idx}: {e}")
                conn.commit()
                continue
            else:
                conn.rollback()
                conn.close()
                raise

        df_completed = _ensure_label_columns(df_completed)
        df_completed = df_completed.replace({"": None})

        # AFTER stats
        after = _count_missing(df_completed)
        filled_sub = before["sub_missing"] - after["sub_missing"]
        filled_obj = before["obj_missing"] - after["obj_missing"]
        total_filled_sub += max(filled_sub, 0)
        total_filled_obj += max(filled_obj, 0)

        print(f"[{landscape}] AFTER   → "
              f"sub_missing={after['sub_missing']}, "
              f"obj_missing={after['obj_missing']}, "
              f"both_missing={after['both_missing']}")
        print(f"[{landscape}] FILLED  → "
              f"subject={max(filled_sub, 0)}, object={max(filled_obj, 0)}")

        # Apply label update rules row-by-row
        conflicts_this_chunk = 0

        for _, row in df_completed.iterrows():
            sid = row.get("subject_id")
            oid = row.get("object_id")
            semra_sub = row.get("subject_label")
            semra_obj = row.get("object_label")

            # Existing labels as seen in DB snapshot (df_chunk)
            existing_sub = row.get("subject_label")  # after SemRA; we need DB snapshot
            existing_obj = row.get("object_label")

            # To respect skip_existing_labels, we need original DB labels.
            # df_chunk had them before SemRA mutated; so we re-align from df_chunk via index.
            # A simpler approach is: treat labels in df_chunk as DB labels.
            # Here, we use df_chunk.loc[row.name] to get them.
            orig = df_chunk.loc[row.name] if row.name in df_chunk.index else None
            if orig is not None:
                existing_sub = orig.get("subject_label")
                existing_obj = orig.get("object_label")

            # SUBJECT SIDE
            if sid:
                if existing_sub is None and semra_sub is not None:
                    # Missing → fill: accept SemRA label, propagate everywhere
                    update_label_everywhere(
                        conn,
                        epairs_table,
                        concept_table,
                        priority_table,
                        sid,
                        semra_sub,
                    )
                elif existing_sub is not None and semra_sub is not None and semra_sub != existing_sub:
                    # Conflict: do NOT overwrite, just record conflict pairs
                    _insert_conflict_pairs(cur, epairs_table, sid, existing_sub, semra_sub)
                    conflicts_this_chunk += 1

            # OBJECT SIDE
            if oid:
                if existing_obj is None and semra_obj is not None:
                    update_label_everywhere(
                        conn,
                        epairs_table,
                        concept_table,
                        priority_table,
                        oid,
                        semra_obj,
                    )
                elif existing_obj is not None and semra_obj is not None and semra_obj != existing_obj:
                    _insert_conflict_pairs(cur, epairs_table, oid, existing_obj, semra_obj)
                    conflicts_this_chunk += 1

        total_conflicts += conflicts_this_chunk
        print(f"[{landscape}] CONFLICTS this chunk: {conflicts_this_chunk}")

        conn.commit()

    # Final remaining unenriched rows for this landscape
    remaining = cur.execute(
        f"SELECT COUNT(*) FROM {priority_table} "
        f"WHERE landscape=? AND added_to_enrich=0",
        (landscape,),
    ).fetchone()[0]

    print(f"\n=== Enrichment complete: {landscape} ===")
    print(f"Total subject labels filled : {total_filled_sub}")
    print(f"Total object labels filled  : {total_filled_obj}")
    print(f"Total conflicts recorded    : {total_conflicts}")
    print(f"Remaining unenriched rows   : {remaining}")
    print("========================================\n")

    conn.close()


def run_semra_enrichment(cfg: PrepConfig, headers: Dict[str, Any]) -> None:
    """
    Top-level entrypoint for SemRA enrichment.

    Called from prep.py after:
      - DB created
      - headers loaded
      - ingestion complete
    """
    if cfg.pipeline.get("skip_semra_enrichment", False):
        print("[PIPELINE] SemRA enrichment skipped via config.")
        return

    print("\n[PIPELINE] Starting SemRA enrichment phase...")

    for landscape in cfg.landscapes:
        enrich_landscape(cfg, headers, landscape)

    print("[PIPELINE] SemRA enrichment phase complete.")


# ============================================================
# UNIT TEST
# ============================================================

if __name__ == "__main__":
    from config_loader import load_config
    from header_utils import load_headers

    cfg = load_config()
    headers = load_headers(cfg)

    run_semra_enrichment(cfg, headers)
