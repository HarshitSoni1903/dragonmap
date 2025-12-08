# src/header_loader.py

import gzip
import pickle
from pathlib import Path
import yaml

# Load config
CONFIG = yaml.safe_load(open("../config/data_prep.yaml"))
DATA_DIR = Path(CONFIG["paths"]["data_dir"])
CACHE_PATH = Path(CONFIG["paths"]["header_cache"])
LANDSCAPES = CONFIG["landscapes"]


def peek_tsv_header(path: Path):
    """Read the header from a .tsv.gz file."""
    with gzip.open(path, "rt") as f:
        return f.readline().strip().split("\t")


def build_header_cache():
    """
    Read all priority headers and store them in
    headers_cache.pkl as a dict:
        landscape → [columns]
    """

    header_map = {}

    for landscape in LANDSCAPES:
        pattern = f"{landscape}_priority*.tsv.gz"
        files = list(DATA_DIR.glob(pattern))

        if not files:
            print(f"[WARNING] No priority files found for {landscape}")
            continue

        header = peek_tsv_header(files[0])
        header_map[landscape] = header

    print(f"[INFO] Saving header cache → {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(header_map, f)

    return header_map


def load_headers():
    """
    Load header cache if present, else rebuild it.
    """
    if CACHE_PATH.exists():
        print(f"[INFO] Loading cached headers from {CACHE_PATH}")
        return pickle.load(open(CACHE_PATH, "rb"))

    print("[INFO] Header cache missing — rebuilding...")
    return build_header_cache()


if __name__ == "__main__":
    headers = load_headers()
    print("\n=== HEADER CACHE ===")
    for lan, cols in headers.items():
        print(f"{lan}: {len(cols)} columns")
