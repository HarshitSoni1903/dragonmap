# db_utils.py
"""
Database utilities for SeMRA preparation pipeline.

Includes:
 - SQLite connection helpers
 - Table creation
 - Auto-enrichment trigger creation (INSERT + UPDATE)
 - Universal label update engine
 - Concept + priority update helpers
 - Replacement-pair generation (old → new, new → old)
 - Insert helpers for ingestion
 - Fetch helpers for SemRA generator
 - DB inspection utilities
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

from config_loader import PrepConfig


# ============================================================
# CONNECTION
# ============================================================

def get_connection(cfg: PrepConfig) -> sqlite3.Connection:
    """Return SQLite connection with pragmas applied."""
    conn = sqlite3.connect(cfg.db_path)
    cur = conn.cursor()

    # Apply DB pragmas from config
    for key, val in cfg.db_pragmas.items():
        cur.execute(f"PRAGMA {key}={val}")

    conn.commit()
    return conn


# ============================================================
# TABLE CREATION
# ============================================================

def create_tables(cfg: PrepConfig, priority_union: List[str]) -> None:
    """
    Create concept_raw, priority_raw, and enriched_pairs tables.
    """
    conn = get_connection(cfg)
    cur = conn.cursor()

    # -----------------------------
    # concept_raw
    # -----------------------------
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {cfg.tables['concept_raw']} (
            curie_id TEXT PRIMARY KEY,
            label TEXT
        )
    """)

    # -----------------------------
    # priority_raw
    # -----------------------------
    cols_sql = ",\n".join([f'"{col}" TEXT' for col in priority_union])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {cfg.tables['priority_raw']} (
            landscape TEXT,
            added_to_enrich INTEGER DEFAULT 0,
            {cols_sql}
        )
    """)

    # -----------------------------
    # enriched_pairs
    # -----------------------------
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {cfg.tables['enriched_pairs']} (
            subject_id TEXT,
            subject_label TEXT,
            object_id TEXT,
            object_label TEXT
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# TRIGGER CREATION
# ============================================================

def create_auto_enrich_trigger(cfg: PrepConfig) -> None:
    """
    Create triggers that auto-populate enriched_pairs and set added_to_enrich=1
    when BOTH subject_label and object_label are present.

    We install TWO triggers:
      - AFTER INSERT (row inserted with both labels already filled)
      - AFTER UPDATE OF subject_label, object_label (row gets labels later via SemRA / backfill / manual updates)
    """
    conn = get_connection(cfg)
    cur = conn.cursor()

    table = cfg.tables["priority_raw"]
    epairs = cfg.tables["enriched_pairs"]

    # Drop any legacy trigger
    cur.execute("DROP TRIGGER IF EXISTS trg_priority_auto_enrich")
    cur.execute("DROP TRIGGER IF EXISTS trg_priority_auto_enrich_insert")
    cur.execute("DROP TRIGGER IF EXISTS trg_priority_auto_enrich_update")

    # -----------------------------
    # INSERT trigger:
    # fire when row is inserted with both labels present
    # -----------------------------
    #
    # cur.execute(f"""
    # CREATE TRIGGER trg_priority_auto_enrich_insert
    # AFTER INSERT ON {table}
    # WHEN NEW.subject_label IS NOT NULL
    #   AND NEW.object_label IS NOT NULL
    #   AND NEW.added_to_enrich = 0
    # BEGIN
    #     INSERT INTO {epairs}(subject_id, subject_label, object_id, object_label)
    #     VALUES (NEW.subject_id, NEW.subject_label, NEW.object_id, NEW.object_label);
    #     UPDATE {table}
    #     SET added_to_enrich = 1
    #     WHERE rowid = NEW.rowid;
    # END;
    # """)

    cur.execute(f"""
        CREATE TRIGGER trg_auto_enrich_insert
        AFTER INSERT ON {table}
        WHEN NEW.subject_label IS NOT NULL
         AND NEW.subject_label <> ''
         AND NEW.object_label IS NOT NULL
         AND NEW.object_label <> ''
         AND NEW.added_to_enrich = 0
        BEGIN
            INSERT INTO {epairs}(subject_id, subject_label, object_id, object_label)
            VALUES (NEW.subject_id, NEW.subject_label, NEW.object_id, NEW.object_label);
            UPDATE {table}
            SET added_to_enrich = 1
            WHERE rowid = NEW.rowid;
        END;
    """)

    # -----------------------------
    # UPDATE trigger:
    # fire when labels become non-null via UPDATE
    # -----------------------------
    #
    # cur.execute(f"""
    # CREATE TRIGGER trg_priority_auto_enrich_update
    # AFTER UPDATE OF subject_label, object_label ON {table}
    # WHEN NEW.subject_label IS NOT NULL
    #   AND NEW.object_label IS NOT NULL
    #   AND NEW.added_to_enrich = 0
    # BEGIN
    #     INSERT INTO {epairs}(subject_id, subject_label, object_id, object_label)
    #     VALUES (NEW.subject_id, NEW.subject_label, NEW.object_id, NEW.object_label);
    #     UPDATE {table}
    #     SET added_to_enrich = 1
    #     WHERE rowid = NEW.rowid;
    # END;
    # """)

    cur.execute(f"""
        CREATE TRIGGER trg_auto_enrich_update
        AFTER UPDATE OF subject_label, object_label ON priority_raw
        WHEN NEW.subject_label IS NOT NULL
         AND NEW.subject_label <> ''
         AND NEW.object_label IS NOT NULL
         AND NEW.object_label <> ''
         AND NEW.added_to_enrich = 0
        BEGIN
            INSERT INTO enriched_pairs(subject_id, subject_label, object_id, object_label)
            VALUES (NEW.subject_id, NEW.subject_label, NEW.object_id, NEW.object_label);
            UPDATE priority_raw
            SET added_to_enrich = 1
            WHERE rowid = NEW.rowid;
        END;
    """)

    conn.commit()
    conn.close()


# ============================================================
# INGESTION HELPERS
# ============================================================

def insert_concept_raw(conn: sqlite3.Connection, curie_id: str, label: Optional[str]) -> None:
    """Insert concept_raw row (used only during ingestion)."""
    conn.execute(
        "INSERT OR IGNORE INTO concept_raw(curie_id, label) VALUES (?, ?)",
        (curie_id, label)
    )


def insert_priority_raw(conn: sqlite3.Connection, table: str,
                        landscape: str, rowdict: Dict[str, Any]) -> None:
    """
    Insert one priority row.
    rowdict must include all priority_union columns.
    """
    cols = ["landscape", "added_to_enrich"] + list(rowdict.keys())
    vals = [landscape, 0] + list(rowdict.values())
    placeholders = ",".join(["?"] * len(vals))
    col_sql = ",".join([f'"{c}"' for c in cols])

    conn.execute(
        f"INSERT INTO {table}({col_sql}) VALUES ({placeholders})",
        vals
    )


# ============================================================
# UNIVERSAL LABEL UPDATE ENGINE
# ============================================================

def _ruleA_should_update(old: Optional[str], new: Optional[str]) -> bool:
    """Rule A: longest non-null wins."""
    if new is None:
        return False
    if old is None:
        return True
    return len(new) > len(old)


def update_label_everywhere(conn: sqlite3.Connection,
                            epairs_table: str,
                            concept_table: str,
                            priority_table: str,
                            curie_id: str,
                            new_label: Optional[str]) -> None:
    """
    Apply longest-label-wins rule to:
      - concept_raw.label
      - priority_raw.subject_label
      - priority_raw.object_label

    Whenever a replacement occurs (short → long):
        Insert BOTH (old→new) AND (new→old) into enriched_pairs.

    NOTE: Updates on priority_raw.*label* will fire the UPDATE trigger and therefore
          may mark rows as enriched if both labels become non-null.
    """
    cur = conn.cursor()

    # -----------------------------
    # 1. Update concept_raw
    # -----------------------------
    cur.execute(f"SELECT label FROM {concept_table} WHERE curie_id=?", (curie_id,))
    row = cur.fetchone()
    old_label = row[0] if row else None

    if old_label is None and new_label is not None:
        # insert or update
        cur.execute(f"""
            INSERT INTO {concept_table}(curie_id, label)
            VALUES (?, ?)
            ON CONFLICT(curie_id) DO UPDATE SET label=excluded.label
        """, (curie_id, new_label))

    elif _ruleA_should_update(old_label, new_label):
        # record enriched pair A→B and B→A
        cur.execute(f"""
            INSERT INTO {epairs_table} (subject_id, subject_label, object_id, object_label)
            VALUES (?, ?, ?, ?)
        """, (curie_id, old_label, curie_id, new_label))

        cur.execute(f"""
            INSERT INTO {epairs_table} (subject_id, subject_label, object_id, object_label)
            VALUES (?, ?, ?, ?)
        """, (curie_id, new_label, curie_id, old_label))

        # update concept
        cur.execute(
            f"UPDATE {concept_table} SET label=? WHERE curie_id=?",
            (new_label, curie_id)
        )

    # -----------------------------
    # 2. Update priority_raw subject_label
    # -----------------------------
    cur.execute(
        f"SELECT subject_label FROM {priority_table} WHERE subject_id=?",
        (curie_id,)
    )
    rows = cur.fetchall()

    for (old_sub,) in rows:
        if _ruleA_should_update(old_sub, new_label):
            # replacement pair
            cur.execute(f"""
                INSERT INTO {epairs_table} (subject_id, subject_label, object_id, object_label)
                VALUES (?, ?, ?, ?)
            """, (curie_id, old_sub, curie_id, new_label))

            cur.execute(f"""
                INSERT INTO {epairs_table} (subject_id, subject_label, object_id, object_label)
                VALUES (?, ?, ?, ?)
            """, (curie_id, new_label, curie_id, old_sub))

            # update field (fires UPDATE trigger if both labels now set)
            cur.execute(
                f"UPDATE {priority_table} SET subject_label=? WHERE subject_id=?",
                (new_label, curie_id)
            )

    # -----------------------------
    # 3. Update priority_raw object_label
    # -----------------------------
    cur.execute(
        f"SELECT object_label FROM {priority_table} WHERE object_id=?",
        (curie_id,)
    )
    rows = cur.fetchall()

    for (old_obj,) in rows:
        if _ruleA_should_update(old_obj, new_label):
            # replacement pairs
            cur.execute(f"""
                INSERT INTO {epairs_table} (subject_id, subject_label, object_id, object_label)
                VALUES (?, ?, ?, ?)
            """, (curie_id, old_obj, curie_id, new_label))

            cur.execute(f"""
                INSERT INTO {epairs_table} (subject_id, subject_label, object_id, object_label)
                VALUES (?, ?, ?, ?)
            """, (curie_id, new_label, curie_id, old_obj))

            # update field (fires UPDATE trigger if both labels now set)
            cur.execute(
                f"UPDATE {priority_table} SET object_label=? WHERE object_id=?",
                (new_label, curie_id)
            )

    conn.commit()


# ============================================================
# FETCH + MARK HELPERS
# ============================================================

def fetch_rows_to_enrich(conn: sqlite3.Connection,
                         table: str, limit: int) -> List[Dict[str, Any]]:
    """Return unenriched rows up to limit."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    result = cur.execute(
        f"SELECT * FROM {table} WHERE added_to_enrich=0 LIMIT ?",
        (limit,)
    ).fetchall()
    return [dict(r) for r in result]


# ============================================================
# CREATE INDEXES
# ============================================================

def create_ingestion_indexes(cfg: PrepConfig) -> None:
    """
    Create all required indexes for fast lookups.
    Call ONLY after first-run ingestion completes.
    Safe to call multiple times (idempotent).
    """
    conn = get_connection(cfg)
    cur = conn.cursor()

    pr = cfg.tables["priority_raw"]
    cr = cfg.tables["concept_raw"]

    print("[DB] Creating ingestion indexes...")

    # Required for update_label_everywhere()
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pr_sub ON {pr}(subject_id)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pr_obj ON {pr}(object_id)")

    # Required for SemRA enrichment / concept lookups
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cr_curie ON {cr}(curie_id)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_cr_label ON {cr}(label)")

    conn.commit()
    conn.close()
    print("[DB] All ingestion indexes created successfully.")


# ============================================================
# INSPECTION
# ============================================================

def inspect_db(cfg: PrepConfig) -> None:
    """Print all tables and their columns."""
    conn = get_connection(cfg)
    cur = conn.cursor()

    print("=== DB Tables ===")
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for (t,) in tables:
        print(f"\nTable: {t}")
        cols = cur.execute(
            f"PRAGMA table_info({t})"
        ).fetchall()

        for col in cols:
            print(" ", col)

    conn.close()


def count_enriched_pairs(cfg: PrepConfig) -> int:
    """
    Return the number of rows in enriched_pairs and print it nicely.
    Safe to call at any time.
    """
    conn = get_connection(cfg)
    cur = conn.cursor()

    table = cfg.tables["enriched_pairs"]
    result = cur.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"[DB] enriched_pairs count = {result}")

    conn.close()
    return result


# ============================================================
# INITIAL BACKFILL
# ============================================================

def initial_backfill_labels(cfg: PrepConfig,
                            concept_table, priority_table, epairs_table):
    """
    One-time initialization pass:
      1. Iterate all concept_raw rows with labels
      2. Call update_label_everywhere to propagate concept labels into
         priority_raw subject/object labels
      3. Triggers on priority_raw will mark rows enriched when both
         subject_label and object_label become non-null.

    Must be run AFTER:
      - concept_raw fully ingested
      - priority_raw fully ingested
      - indexes created
      - create_auto_enrich_trigger(cfg) has been called
    """
    conn = get_connection(cfg)
    cur = conn.cursor()

    # ---- 1. Fetch all known concept labels ----
    rows = cur.execute(
        f"SELECT curie_id, label FROM {concept_table} WHERE label IS NOT NULL"
    ).fetchall()

    print(f"[INIT] Found {len(rows)} concepts with labels")

    # ---- 2. Apply rule A everywhere ----
    count = 0
    for curie_id, label in rows:
        update_label_everywhere(
            conn, epairs_table, concept_table,
            priority_table, curie_id, label
        )
        count += 1
        if count % 10000 == 0:
            print(f"[INIT] Processed {count} concept IDs...")

    conn.commit()
    count_enriched_pairs(cfg)
    conn.commit()
    conn.close()

    print("[INIT] Initial label backfill completed.")
