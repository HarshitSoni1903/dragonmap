# header_utils.py
"""
Header loader for priority TSV files only.

Rules:
 - If header_cache.pkl exists → load it
 - Otherwise:
       - Read priority TSV headers for each landscape
       - Build priority_headers and priority_union
       - Save cache
"""

import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from config_loader import PrepConfig, load_config


# -------------------------
# Helpers
# -------------------------

def _read_header(path: Path) -> Optional[List[str]]:
    """Return the first-line header of a .tsv.gz file."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.readline().rstrip("\n").split("\t")
    except Exception as e:
        print(f"[WARNING] Could not read {path}: {e}")
        return None


def _glob_files(data_dir: Path, landscape: str, pattern: str) -> List[Path]:
    """Glob files like anatomy_priority*.tsv.gz."""
    return sorted(data_dir.glob(pattern.format(landscape=landscape)))


# -------------------------
# Main Header Loader
# -------------------------

def load_headers(cfg: PrepConfig) -> Dict[str, Dict[str, List[str]]]:
    """
    Load headers for priority TSVs.

    Output dict structure:
    {
        "priority_headers": { landscape: [column names] },
        "priority_union":   [sorted unique columns across all landscapes]
    }
    """

    cache = cfg.header_cache

    # 1. If cache exists → use it
    if cache.exists():
        print(f"[INFO] Using cached headers → {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("[INFO] No header cache found. Extracting priority headers...")

    priority_headers: Dict[str, List[str]] = {}
    priority_pattern = "{landscape}_priority*.tsv.gz"

    # read priority headers for each landscape
    for landscape in cfg.landscapes:
        files = _glob_files(cfg.data_dir, landscape, priority_pattern)
        if not files:
            print(f"[WARNING] No priority files found for '{landscape}'")
            continue

        header = _read_header(files[0])
        if header:
            priority_headers[landscape] = header

    # 2. Compute union
    priority_union = sorted({c for cols in priority_headers.values() for c in cols})

    out = {
        "priority_headers": priority_headers,
        "priority_union": priority_union,
    }

    # 3. Cache
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(out, f)
        print(f"[INFO] Header cache created → {cache}")
    except Exception as e:
        print(f"[WARNING] Failed to write header cache: {e}")

    return out


# -------------------------
# Test Block
# -------------------------

if __name__ == "__main__":
    cfg = load_config()

    print("[TEST] Running load_headers()...")
    headers = load_headers(cfg)

    print("\n[TEST] Headers loaded:")
    print("Priority headers:", list(headers["priority_headers"].keys()))
    print("Union columns:", len(headers["priority_union"]))
