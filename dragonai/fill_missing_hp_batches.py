import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ----------------- Paths -----------------
DB = os.path.expanduser("~/Desktop/curategpt/stagedb")
IN = os.path.expanduser("~/Desktop/hp_enriched.csv")
MISS = os.path.expanduser("~/Desktop/hp_missing.csv")
GEN = os.path.expanduser("~/Desktop/hp_generated_partial.csv")  # checkpoint
OUT = os.path.expanduser("~/Desktop/hp_enriched_filled.csv")  # final merged
COST = os.path.expanduser("~/Desktop/hp_fill_cost_log.csv")  # per-row costs

# ----------------- Pricing / tokens -----------------
IN_PRICE = float(os.getenv("LLM_IN_PRICE_PER_1K", "0.00015"))
OUT_PRICE = float(os.getenv("LLM_OUT_PRICE_PER_1K", "0.00060"))
ASSUMED_PROMPT_TOKENS = int(os.getenv("ASSUMED_PROMPT_TOKENS", "350"))

# ----------------- Parallel / batching params -----------------
BATCH_SIZE = 20  # number of labels per batch
MAX_WORKERS = 5  # how many calls in parallel inside each batch
BATCH_SAVE_EVERY = 2  # save checkpoint every N completed batches
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.3  # small pause per label (in worker)


# ----------------- Helpers -----------------
def run(cmd: str):
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True)


def cgpt_complete(label: str) -> str:
    """Call CurateGPT to complete label/def/synonyms for a given label."""
    cmd = f'poetry run curategpt complete -p "{DB}" -c ont_hp "{label}"'
    for attempt in range(1, MAX_RETRIES + 1):
        p = run(cmd)
        txt = (p.stdout or "").strip()
        if p.returncode == 0 and txt:
            return txt
        time.sleep(0.6 * attempt)
    return ""


def extract_section(txt: str, hdr: str) -> str:
    m = re.search(
        rf"(?im)^{hdr}\s*:\s*\n?(.*?)(?=\n\s*\n|\n[A-Za-z][^\n]*:\s*$|\Z)",
        txt,
        flags=re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def first_sentence(block: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (block or "").strip())
    return parts[0].strip() if parts and parts[0].strip() else ""


def norm_synonyms(block: str) -> str:
    if not block:
        return ""
    lines = [ln.strip(" -•\t") for ln in block.splitlines() if ln.strip()]
    items = []
    for ln in lines:
        items += [p.strip() for p in re.split(r"[;|,]", ln) if p.strip()]
    seen, out = set(), []
    for w in items:
        k = w.lower()
        if k not in seen:
            seen.add(k)
            out.append(w)
    return " | ".join(out)


def extract_definition(txt: str) -> str:
    b = extract_section(txt, "definition")
    return b if b else first_sentence(txt)


# def extract_synonyms(txt: str) -> str:
#     b = extract_section(txt, "synonyms?")
#     return norm_synonyms(b)


def extract_synonyms(txt: str) -> str:
    # 1. Prefer an explicit "synonyms:" block if CurateGPT ever adds one
    b = extract_section(txt, "synonyms?")
    # 2. Fallback: use "aliases:" (what CurateGPT currently outputs)
    if not b:
        b = extract_section(txt, "aliases?")
    return norm_synonyms(b)


def rough_tokens(text: str) -> int:
    return max(1, len((text or "")) // 4)


def process_label(label: str):
    """Worker for a single label; returns (row, cost_row)."""
    txt = cgpt_complete(label)
    gdef = extract_definition(txt)
    gsyn = extract_synonyms(txt)

    row = {
        "label": label,
        "generated_definition": gdef,
        "generated_synonyms": gsyn,
        "raw": txt,
    }

    out_tokens = rough_tokens(txt)
    in_tokens = ASSUMED_PROMPT_TOKENS

    cost_row = {
        "label": label,
        "assumed_input_tokens": in_tokens,
        "output_tokens_est": out_tokens,
        "input_cost_usd": round((in_tokens / 1000) * IN_PRICE, 6),
        "output_cost_usd": round((out_tokens / 1000) * OUT_PRICE, 6),
        "total_cost_usd": round(
            (in_tokens / 1000) * IN_PRICE + (out_tokens / 1000) * OUT_PRICE,
            6,
        ),
    }

    time.sleep(SLEEP_BETWEEN)
    return row, cost_row


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ----------------- Load base data -----------------
df_all = pd.read_csv(IN)
df_miss = pd.read_csv(MISS)

# ----------------- Resume state -----------------
done_labels = set()
rows = []

if os.path.exists(GEN):
    prev = pd.read_csv(GEN)
    if not prev.empty:
        rows.extend(prev.to_dict("records"))
        if "label" in prev.columns:
            done_labels.update(prev["label"].tolist())
        print(f"[resume] Loaded {len(prev)} previously generated rows.")

miss_labels = [lbl for lbl in df_miss["label"].tolist() if lbl not in done_labels]
print(f"Missing to generate now: {len(miss_labels)} (already done: {len(done_labels)})")

# Cost log
if os.path.exists(COST):
    cost_df = pd.read_csv(COST)
    print(f"[resume] Loaded existing cost log with {len(cost_df)} rows.")
else:
    cost_df = pd.DataFrame(
        columns=[
            "label",
            "assumed_input_tokens",
            "output_tokens_est",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
        ]
    )

# ----------------- Batched parallel processing -----------------
start_time = time.time()
completed_rows = 0
completed_batches = 0

for batch_idx, batch_labels in enumerate(chunks(miss_labels, BATCH_SIZE), start=1):
    if not batch_labels:
        continue

    print(f"[batch {batch_idx}] processing {len(batch_labels)} labels...")
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch_labels))) as ex:
        future_to_label = {ex.submit(process_label, lbl): lbl for lbl in batch_labels}

        for fut in as_completed(future_to_label):
            lbl = future_to_label[fut]
            try:
                row, cost_row = fut.result()
            except Exception as e:
                print(f"[error] label {lbl}: {e}")
                continue

            rows.append(row)
            cost_df = pd.concat([cost_df, pd.DataFrame([cost_row])], ignore_index=True)
            completed_rows += 1

    completed_batches += 1

    # Checkpoint after certain number of batches
    if completed_batches % BATCH_SAVE_EVERY == 0:
        pd.DataFrame(rows).to_csv(GEN, index=False)
        cost_df.to_csv(COST, index=False)
        elapsed = time.time() - start_time
        print(
            f"[checkpoint] batches: {completed_batches}, "
            f"rows: {completed_rows}, "
            f"elapsed {elapsed / 60:.1f} min"
        )
        sys.stdout.flush()

# ----------------- Final save / timing -----------------
pd.DataFrame(rows).to_csv(GEN, index=False)
cost_df.to_csv(COST, index=False)

elapsed = time.time() - start_time
print(
    f"[done] generated rows: {len(rows)} | cost rows: {len(cost_df)} | "
    f"elapsed: {elapsed / 60:.2f} minutes (~{elapsed:.0f} seconds)"
)

# ----------------- Merge back into enriched table -----------------
df_gen = pd.DataFrame(rows)

merged = df_all.merge(
    df_gen[["label", "generated_definition", "generated_synonyms"]],
    on="label",
    how="left",
)

merged["final_definition"] = merged["definition"].fillna("")
need_def = merged["final_definition"].str.len() == 0
merged.loc[need_def, "final_definition"] = merged["generated_definition"].fillna("")

merged["final_synonyms"] = merged["synonyms"].fillna("")
need_syn = merged["final_synonyms"].str.len() == 0
merged.loc[need_syn, "final_synonyms"] = merged["generated_synonyms"].fillna("")

merged["missing_definition_after_fill"] = merged["final_definition"].str.len() == 0
merged["missing_synonyms_after_fill"] = merged["final_synonyms"].str.len() == 0

merged.to_csv(OUT, index=False)

print("Wrote final:", OUT)
print(
    "Remaining missing -> definition:",
    int(merged["missing_definition_after_fill"].sum()),
    " | synonyms:",
    int(merged["missing_synonyms_after_fill"].sum()),
)

print("Cost log:", COST)
print("Estimated total cost so far: $", round(cost_df["total_cost_usd"].sum(), 4))
