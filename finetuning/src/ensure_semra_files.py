import sys
import time
from pathlib import Path
from typing import Dict

import requests

from config_loader import load_config


ZENODO_RECORDS: Dict[str, str] = {
    "disease":  "15826693",
    "anatomy":  "15826754",
    "gene":     "15826794",
    "cell_line":"15826779",
}


def build_zenodo_url(record_id: str, filename: str) -> str:
    """
    Build a direct download URL for a given Zenodo record + filename.
    """
    return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def download_file(url: str, dst_path: Path, chunk_size: int = 8192) -> None:
    """
    Stream-download 'url' to 'dst_path'.
    Skips if dst_path already exists (handled by caller).
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] GET {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

    total = int(resp.headers.get("content-length", 0)) or None
    downloaded = 0
    t0 = time.time()

    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total is not None:
                pct = downloaded / total * 100
                sys.stdout.write(f"\r[DOWNLOAD] {dst_path.name}: {pct:6.2f}%")
                sys.stdout.flush()

    if total is not None:
        sys.stdout.write("\n")

    tmp_path.replace(dst_path)
    dt = time.time() - t0
    print(f"[DOWNLOAD] Saved to {dst_path} in {dt:.1f}s\n")


def ensure_semra_files():
    """
    For each landscape, ensure we have:
      {data_dir}/{landscape}_concept/concept_nodes.tsv.gz
      {data_dir}/{landscape}_priority/priority.sssom.tsv.gz

    `priority.sssom.tsv.gz` is downloaded from Zenodo's `processed.sssom.tsv.gz`.
    """
    cfg = load_config()
    data_dir = Path(cfg.data_dir).expanduser().resolve()
    print(f"[INFO] Using data_dir = {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    for landscape, record_id in ZENODO_RECORDS.items():
        print(f"\n===== {landscape.upper()} =====")

        concept_dir = data_dir
        priority_dir = data_dir

        concept_path = concept_dir / f"{landscape}_concept_nodes.tsv.gz"
        priority_path = priority_dir / f"{landscape}_priority.sssom.tsv.gz"

        # concept_nodes.tsv.gz
        if concept_path.exists():
            print(f"[SKIP] {concept_path} already exists.")
        else:
            concept_url = build_zenodo_url(record_id, "concept_nodes.tsv.gz")
            download_file(concept_url, concept_path)

        # processed.sssom.tsv.gz -> saved as priority.sssom.tsv.gz
        if priority_path.exists():
            print(f"[SKIP] {priority_path} already exists.")
        else:
            processed_url = build_zenodo_url(record_id, "processed.sssom.tsv.gz")
            download_file(processed_url, priority_path)

    print("\n[INFO] All requested SeMRA files are present.")


if __name__ == "__main__":
    ensure_semra_files()
