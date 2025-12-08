# export_utils.py
"""
Export final positive training pairs for SapBERT.

Exports 4 CSV files:
 - sapbert_all.csv
 - sapbert_train.csv
 - sapbert_val.csv
 - sapbert_test.csv

Filtering rules:
  * Remove rows where subject_label == object_label (after normalization)
  * Remove exact duplicates (after normalization)
  * Keep reversed pairs (A,B) and (B,A)
  * Normalize only for dedup + identity filtering

Splitting:
  * Random seed for reproducibility
  * Train/Val/Test = 80 / 10 / 10
"""

from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List

from config_loader import PrepConfig
from db_utils import get_connection


# ============================================================
# Normalization
# ============================================================

def normalize_text(s: str) -> str:
    """Lowercase, strip, collapse whitespace. Return '' for None."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# Chunked DB loader
# ============================================================

def load_enriched_pairs_chunked(cfg: PrepConfig) -> pd.DataFrame:
    """
    Generator yielding dataframes of enriched pairs in chunks.

    Columns expected in enriched_pairs:
      subject_label, object_label
    """
    conn = get_connection(cfg)
    cur = conn.cursor()

    epairs_table = cfg.tables["enriched_pairs"]
    total_rows = cur.execute(f"SELECT COUNT(*) FROM {epairs_table}").fetchone()[0]

    offset = 0
    chunksize = cfg.chunk_size

    while True:
        df = pd.read_sql_query(
            f"""
            SELECT subject_label, object_label
            FROM {epairs_table}
            ORDER BY rowid
            LIMIT {chunksize} OFFSET {offset};
            """,
            conn,
        )

        if df.empty:
            break

        yield df
        offset += chunksize

    conn.close()


# ============================================================
# Export logic
# ============================================================

def export_sapbert_pairs(cfg: PrepConfig) -> None:
    """
    Export enriched_pairs → sapbert_all.csv, train/val/test splits.
    Applies normalization-based filtering & dedupe.
    """

    print("[EXPORT] Starting SapBERT export...")

    all_rows = []

    # --------------------------
    # 1. Load in chunks
    # --------------------------
    for df in load_enriched_pairs_chunked(cfg):
        for _, row in df.iterrows():
            sub = row["subject_label"]
            obj = row["object_label"]

            # drop None / empty
            if not sub or not obj:
                continue

            nsub = normalize_text(sub)
            nobj = normalize_text(obj)

            # drop identity pairs
            if nsub == nobj:
                continue

            # we keep reversed pairs, but dedupe exact duplicates
            all_rows.append((sub, obj, nsub, nobj))

    if not all_rows:
        print("[EXPORT] No valid enrichment pairs found. Nothing to export.")
        return

    print(f"[EXPORT] Loaded {len(all_rows)} raw pairs before dedupe.")

    # --------------------------
    # 2. Dedupe
    # --------------------------
    seen_keys = set()
    final_pairs = []

    for sub, obj, nsub, nobj in all_rows:
        key = (nsub, nobj)  # reversed allowed; only exact (nsub,nobj) dropped
        if key not in seen_keys:
            seen_keys.add(key)
            final_pairs.append((sub, obj))

    print(f"[EXPORT] After dedupe: {len(final_pairs)} unique pairs.")

    # --------------------------
    # 3. Convert to DataFrame
    # --------------------------
    df_all = pd.DataFrame(final_pairs, columns=["subject_label", "object_label"])

    # --------------------------
    # 4. Random shuffle + split
    # --------------------------
    SEED = 1337
    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n = len(df_all)
    t_end = int(0.80 * n)
    v_end = int(0.90 * n)

    df_train = df_all.iloc[:t_end]
    df_val   = df_all.iloc[t_end:v_end]
    df_test  = df_all.iloc[v_end:]

    # --------------------------
    # 5. Output paths
    # --------------------------
    out_all  = Path(cfg.data_dir) / "sapbert_all.csv"
    out_train = Path(cfg.data_dir) / "sapbert_train.csv"
    out_val   = Path(cfg.data_dir) / "sapbert_val.csv"
    out_test  = Path(cfg.data_dir) / "sapbert_test.csv"

    # --------------------------
    # 6. Write files
    # --------------------------
    df_all.to_csv(out_all, index=False)
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"[EXPORT] sapbert_all.csv   → {out_all}")
    print(f"[EXPORT] sapbert_train.csv → {out_train}")
    print(f"[EXPORT] sapbert_val.csv   → {out_val}")
    print(f"[EXPORT] sapbert_test.csv  → {out_test}")

    print("[EXPORT] Done.")


# ============================================================
# Unit test
# ============================================================

if __name__ == "__main__":
    from config_loader import load_config
    cfg = load_config()
    export_sapbert_pairs(cfg)
