import argparse
import semra
from semra.api import prioritize
import os
import pandas as pd


# ---------------------------------------------------------
# EXTRACT LABEL PAIRS FOR ONE LANDSCAPE
# ---------------------------------------------------------

def extract_label_pairs(processed_sssom_path):
    print(f"[INFO] Loading mapping set: {processed_sssom_path}")
    mapping_set = semra.from_sssom(processed_sssom_path)

    print("[INFO] Running prioritization...")
    prioritized = prioritize(mapping_set)

    print("[INFO] Extracting label pairs...")
    pairs = []
    for stmt in prioritized.statements:
        sl = stmt.subject_label
        ol = stmt.object_label

        if sl and ol:
            pairs.append((sl, ol))

    df = pd.DataFrame(pairs, columns=["label1", "label2"])
    print(f"[INFO] Extracted {len(df)} resolved label pairs.")
    return df


# ---------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------

def main(base_dir):
    base_dir = os.path.abspath(base_dir)

    out_dir = os.path.join(base_dir, "outputs")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Landscape paths relative to base_dir
    LANDSCAPE_PATHS = {
        "gene":     os.path.join(base_dir, "gene/processed.sssom.tsv.gz"),
        "disease":  os.path.join(base_dir, "disease/processed.sssom.tsv.gz"),
        "cell":     os.path.join(base_dir, "cell_line/processed.sssom.tsv.gz"),
        "anatomy":  os.path.join(base_dir, "anatomy/processed.sssom.tsv.gz"),
    }

    all_frames = []

    for name, path in LANDSCAPE_PATHS.items():
        try:
            print(f"\n[PROCESSING LANDSCAPE]: {name}")
            df = extract_label_pairs(path)

            df["landscape"] = name
            outfile = os.path.join(out_dir, f"{name}_pairs.csv")
            df.to_csv(outfile, index=False)

            print(f"[INFO] Saved: {outfile}")
            all_frames.append(df)

        except Exception as e:
            errfile = os.path.join(log_dir, f"{name}_error.txt")
            with open(errfile, "w") as f:
                f.write(str(e))
            print(f"[ERROR] Failed for {name}. Logged in {errfile}")

    # Merge everything
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True).drop_duplicates()
        final_path = os.path.join(out_dir, "all_pairs.csv")
        combined.to_csv(final_path, index=False)

        print(f"\n[INFO] Combined dataset: {len(combined)} pairs")
        print(f"[INFO] Saved final combined file to {final_path}")
    else:
        print("\n[ERROR] No landscapes successfully processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="")
    args = parser.parse_args()
    main(args.base_dir)
