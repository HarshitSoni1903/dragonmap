# obo_enrichment.py
"""
Ontology-based enrichment using PyOBO.

Goals:
 - Work over concept_raw in chunks (cfg.chunk_size)
 - For each curie_id, use PyOBO to fetch:
     * preferred label (term.name)
     * synonyms (exact/broad/related)
     * xref labels and synonyms (depth 1)
 - If concept label is missing and OBO has a name → fill it via update_label_everywhere
 - Generate additional positive pairs:
     * main_label ↔ synonym
     * main_label ↔ xref_label
     * synonym ↔ synonym (when available)
 - Do NOT overwrite existing labels when they differ; only add pairs
 - Insert all new pairs into enriched_pairs
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

import pandas as pd

from config_loader import PrepConfig
from db_utils import get_connection, update_label_everywhere

try:
    import pyobo
except ImportError:
    pyobo = None


# ============================================================
# Normalization helper (same logic as export)
# ============================================================

def _normalize_text(s: Optional[str]) -> str:
    """Lowercase, strip, collapse whitespace. Return '' for None."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# Concept chunk loader
# ============================================================

def _load_concept_chunks(cfg: PrepConfig):
    """
    Yield chunks from concept_raw with columns (curie_id, label).

    Uses OFFSET-based paging; safe because we only update labels, not row count.
    """
    conn = get_connection(cfg)
    concept_table = cfg.tables["concept_raw"]

    offset = 0
    chunksize = cfg.chunk_size

    while True:
        df = pd.read_sql_query(
            f"""
            SELECT curie_id, label
            FROM {concept_table}
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
# OBO label + synonym fetch
# ============================================================

def _split_curie(curie_id: str) -> Optional[Tuple[str, str]]:
    """Split CURIE like 'MONDO:0005148' into (prefix, identifier)."""
    if ":" not in curie_id:
        return None
    prefix, local_id = curie_id.split(":", 1)
    prefix = prefix.strip()
    local_id = local_id.strip()
    if not prefix or not local_id:
        return None
    return prefix, local_id


def _get_term_and_synonyms(curie_id: str) -> Tuple[Optional[str], List[str]]:
    """
    Use PyOBO to get preferred label + synonyms for a given CURIE.

    Returns:
      (preferred_label, synonyms_list)
    """
    if pyobo is None:
        return None, []

    split = _split_curie(curie_id)
    if split is None:
        return None, []

    prefix, local_id = split

    try:
        term = pyobo.get_term(prefix, local_id)  # type: ignore[attr-defined]
    except Exception:
        return None, []

    if term is None:
        return None, []

    preferred = term.name

    synonyms: List[str] = []
    # term.synonyms is a list of Synonym objects with a .name attribute
    for syn in getattr(term, "synonyms", []) or []:
        name = getattr(syn, "name", None)
        if name:
            synonyms.append(name)

    return preferred, synonyms


def _get_xref_labels_and_synonyms(curie_id: str) -> List[str]:
    """
    Depth-1 xref resolution:
      From the given term, follow xrefs to other CURIEs and collect their
      preferred labels and synonyms.

    Returns list of extra label strings.
    """
    if pyobo is None:
        return []

    split = _split_curie(curie_id)
    if split is None:
        return []

    prefix, local_id = split

    try:
        term = pyobo.get_term(prefix, local_id)  # type: ignore[attr-defined]
    except Exception:
        return []

    if term is None:
        return []

    labels: List[str] = []
    for ref in getattr(term, "xrefs", []) or []:
        ref_prefix = getattr(ref, "prefix", None)
        ref_id = getattr(ref, "identifier", None)
        if not ref_prefix or not ref_id:
            continue

        try:
            ref_term = pyobo.get_term(ref_prefix, ref_id)  # type: ignore[attr-defined]
        except Exception:
            continue
        if ref_term is None:
            continue

        # Preferred label for xref term
        if getattr(ref_term, "name", None):
            labels.append(ref_term.name)  # type: ignore[attr-defined]

        # Synonyms of xref term
        for syn in getattr(ref_term, "synonyms", []) or []:
            name = getattr(syn, "name", None)
            if name:
                labels.append(name)

    return labels


# ============================================================
# Pair insertion
# ============================================================

def _insert_pair(cur, epairs_table: str, label_a: str, label_b: str) -> None:
    """
    Insert a single pair into enriched_pairs as:
      subject_label = label_a, object_label = label_b
    IDs are left as NULL; these are purely text pairs for SapBERT.
    """
    cur.execute(
        f"""
        INSERT INTO {epairs_table}
        (subject_id, subject_label, object_id, object_label)
        VALUES (?, ?, ?, ?)
        """,
        (None, label_a, None, label_b),
    )


# ============================================================
# Main OBO enrichment
# ============================================================

def run_obo_enrichment(cfg: PrepConfig) -> None:
    """
    Perform ontology-based enrichment over concept_raw using PyOBO.

    - Fills missing concept labels when OBO can provide a preferred name
    - Generates extra positive pairs from:
        * preferred label + synonyms
        * preferred label + xref labels (and synonyms)
        * synonym ↔ synonym
    - Does NOT overwrite existing labels; only fills when missing
    - Appends all new pairs to enriched_pairs
    """
    if pyobo is None:
        print("[OBO] PyOBO is not installed; skipping OBO enrichment.")
        return

    print("\n[OBO] Starting OBO-based enrichment...")

    conn = get_connection(cfg)
    cur = conn.cursor()

    concept_table = cfg.tables["concept_raw"]
    priority_table = cfg.tables["priority_raw"]
    epairs_table = cfg.tables["enriched_pairs"]

    chunk_idx = 0
    total_concepts = 0
    total_labels_filled = 0
    total_pairs_added = 0

    for df_chunk in _load_concept_chunks(cfg):
        chunk_idx += 1
        n_chunk = len(df_chunk)
        total_concepts += n_chunk

        print(f"[OBO] Chunk {chunk_idx} → {n_chunk} concepts")

        pairs_this_chunk = 0
        labels_filled_this_chunk = 0

        for _, row in df_chunk.iterrows():
            curie_id = row["curie_id"]
            existing_label = row["label"]

            # --- Fetch OBO labels ---
            preferred, synonyms = _get_term_and_synonyms(curie_id)
            xref_labels = _get_xref_labels_and_synonyms(curie_id)

            # Nothing from OBO → skip
            if preferred is None and not synonyms and not xref_labels:
                continue

            # Decide main label for pair generation
            main_label = existing_label or preferred

            if main_label is None:
                # We still do not know a main label; cannot generate pairs
                continue

            # If label is missing and OBO has a preferred → fill using update_label_everywhere
            if existing_label is None and preferred is not None:
                try:
                    update_label_everywhere(
                        conn,
                        epairs_table,
                        concept_table,
                        priority_table,
                        curie_id,
                        preferred,
                    )
                    labels_filled_this_chunk += 1
                except Exception as e:
                    if cfg.semra_options.get("allow_external_resolution_errors", True):
                        print(f"[OBO] [WARN] Failed to update label for {curie_id}: {e}")
                    else:
                        conn.rollback()
                        conn.close()
                        raise

            # Build a set of all candidate labels around this concept
            candidate_labels: List[str] = [main_label]
            candidate_labels.extend(synonyms)
            candidate_labels.extend(xref_labels)

            # Normalize + filter out obviously bad entries
            cleaned: List[str] = []
            seen_norm = set()

            for lab in candidate_labels:
                if lab is None:
                    continue
                norm = _normalize_text(lab)
                if not norm:
                    continue
                # Skip duplicates at this concept level
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                cleaned.append(lab)

            if len(cleaned) < 2:
                # Need at least two distinct labels to form a pair
                continue

            # Generate pairs for all ordered pairs (i,j), i != j
            # This yields main↔synonym and synonym↔synonym pairs.
            for i in range(len(cleaned)):
                for j in range(len(cleaned)):
                    if i == j:
                        continue
                    a = cleaned[i]
                    b = cleaned[j]

                    # Drop identity after normalization
                    if _normalize_text(a) == _normalize_text(b):
                        continue

                    try:
                        _insert_pair(cur, epairs_table, a, b)
                        pairs_this_chunk += 1
                    except Exception as e:
                        if cfg.semra_options.get("allow_external_resolution_errors", True):
                            print(f"[OBO] [WARN] Failed to insert pair: {e}")
                            continue
                        else:
                            conn.rollback()
                            conn.close()
                            raise

        conn.commit()

        total_pairs_added += pairs_this_chunk
        total_labels_filled += labels_filled_this_chunk

        print(f"[OBO] Chunk {chunk_idx} stats → "
              f"labels_filled={labels_filled_this_chunk}, "
              f"pairs_added={pairs_this_chunk}")

    print("\n[OBO] OBO enrichment complete.")
    print(f"[OBO] Total concepts processed : {total_concepts}")
    print(f"[OBO] Total labels filled      : {total_labels_filled}")
    print(f"[OBO] Total pairs added        : {total_pairs_added}")
    print("========================================\n")

    conn.close()


# ============================================================
# UNIT TEST
# ============================================================

if __name__ == "__main__":
    from config_loader import load_config

    cfg = load_config()
    run_obo_enrichment(cfg)
