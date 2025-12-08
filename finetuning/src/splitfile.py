import pandas as pd
import re
import random
from pathlib import Path

# -----------------------------
# Normalization (same as export_utils)
# -----------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

# -----------------------------
# Concept-level split script
# -----------------------------
def split_concept_level(
    input_csv,
    out_train="data/sapbert_train.csv",
    out_val="data/sapbert_val.csv",
    out_test="data/sapbert_test.csv",
    seed=42,
):
    print("[SPLIT] Loading data...")
    df = pd.read_csv(input_csv)

    # Normalize for concept grouping
    print("[SPLIT] Normalizing concepts...")
    df["nsub"] = df["subject_label"].apply(normalize_text)
    df["nobj"] = df["object_label"].apply(normalize_text)

    # All concepts (from subject side only or both?)
    # SapBERT typically uses SUBJECT concept as anchor for grouping
    concepts = set(df["nsub"].tolist())
    concepts.update(df["nobj"].tolist())

    concepts = list(concepts)
    random.Random(seed).shuffle(concepts)

    n = len(concepts)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train_concepts = set(concepts[:n_train])
    val_concepts = set(concepts[n_train:n_train + n_val])
    test_concepts = set(concepts[n_train + n_val:])

    print(f"[SPLIT] Total concepts: {n}")
    print(f"[SPLIT] Train concepts: {len(train_concepts)}")
    print(f"[SPLIT] Val concepts:   {len(val_concepts)}")
    print(f"[SPLIT] Test concepts:  {len(test_concepts)}")

    # -----------------------------
    # Assign pairs to splits
    # -----------------------------
    train_rows = []
    val_rows = []
    test_rows = []

    for _, row in df.iterrows():
        nsub = row["nsub"]
        nobj = row["nobj"]

        # A pair belongs to the split of its SUBJECT concept
        # (Stable, avoids reversed-pair leakage)
        if nsub in train_concepts:
            train_rows.append(row)
        elif nsub in val_concepts:
            val_rows.append(row)
        elif nsub in test_concepts:
            test_rows.append(row)
        else:
            # If a subject concept somehow falls outside buckets (shouldn't happen)
            # fallback: send to train
            train_rows.append(row)

    # Convert to DataFrame, drop normalization columns
    df_train = pd.DataFrame(train_rows).drop(columns=["nsub", "nobj"])
    df_val   = pd.DataFrame(val_rows).drop(columns=["nsub", "nobj"])
    df_test  = pd.DataFrame(test_rows).drop(columns=["nsub", "nobj"])

    # Save
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print("[SPLIT] Output written:")
    print(f"  Train → {out_train} ({len(df_train)} rows)")
    print(f"  Val   → {out_val}   ({len(df_val)} rows)")
    print(f"  Test  → {out_test}  ({len(df_test)} rows)")


if __name__ == "__main__":
    # Change this to your combined exported file
    split_concept_level("data/sapbert_all.csv")
