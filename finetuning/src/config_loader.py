# config_loader.py

import yaml
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================
# Unified configuration object for the full pipeline
# ============================================================

@dataclass
class PrepConfig:
    # paths
    data_dir: Path
    db_path: Path
    log_file: Path
    header_cache: Path

    # core processing settings
    landscapes: list
    tables: dict
    chunk_size: int

    # logging
    redirect_stdout: bool
    redirect_stderr: bool
    suppress_warnings: bool

    # SemRA options
    semra_options: dict

    # ingestion settings
    ingestion: dict

    # db pragmas
    db_pragmas: dict

    # enrichment settings
    enrichment: dict

    # pipeline step toggles
    pipeline: dict


# ============================================================
# Helper: ensure parent directories exist
# ============================================================

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load YAML → PrepConfig
# ============================================================

def load_config() -> PrepConfig:
    """
    Load YAML config for the SeMRA pipeline, validate fields,
    normalize all relative paths, ensure directory structure, 
    and return a fully populated PrepConfig object.
    """

    yaml_path = Path(__file__).parent.parent / "config" / "prep.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML file does not contain a valid dictionary.")

    # --------------------------------------------------------
    # Extract all sections required for full pipeline execution
    # --------------------------------------------------------
    required_sections = [
        "paths",
        "landscapes",
        "tables",
        "chunking",
        "logging",
        "semra",
        "ingestion",
        "db",
        "enrichment",
        "pipeline",
    ]

    missing = [s for s in required_sections if s not in raw]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    paths_cfg      = raw["paths"]
    chunking_cfg   = raw["chunking"]
    logging_cfg    = raw["logging"]
    semra_cfg      = raw["semra"]
    ingestion_cfg  = raw["ingestion"]
    db_cfg         = raw["db"]
    enrich_cfg     = raw["enrichment"]
    pipeline_cfg   = raw["pipeline"]

    # --------------------------------------------------------
    # Normalize paths (relative to YAML directory)
    # --------------------------------------------------------
    base_dir = yaml_path.parent

    def resolve_rel(p):
        return (base_dir / p).resolve()

    data_dir     = resolve_rel(paths_cfg["data_dir"])
    db_path      = resolve_rel(paths_cfg["db_path"])
    log_file     = resolve_rel(paths_cfg["log_file"])
    header_cache = resolve_rel(paths_cfg["header_cache"])

    # ensure dirs exist
    ensure_parent(db_path)
    ensure_parent(log_file)
    ensure_parent(header_cache)

    # --------------------------------------------------------
    # Extract values into PrepConfig fields
    # --------------------------------------------------------
    cfg = PrepConfig(
        data_dir=data_dir,
        db_path=db_path,
        log_file=log_file,
        header_cache=header_cache,

        landscapes=raw["landscapes"],
        tables=raw["tables"],

        chunk_size=int(chunking_cfg["semra_chunk_size"]),

        # logging
        redirect_stdout=bool(logging_cfg.get("redirect_stdout", True)),
        redirect_stderr=bool(logging_cfg.get("redirect_stderr", True)),
        suppress_warnings=bool(logging_cfg.get("suppress_warnings", True)),

        # SemRA options
        semra_options={
            "enable_label_completion":
                bool(semra_cfg.get("enable_label_completion", True)),
            "allow_external_resolution_errors":
                bool(semra_cfg.get("allow_external_resolution_errors", True)),
        },

        ingestion=ingestion_cfg,
        db_pragmas=db_cfg.get("pragmas", {}),
        enrichment=enrich_cfg,
        pipeline=pipeline_cfg,
    )

    # --------------------------------------------------------
    # OPTIONAL: Short diagnostic summary
    # --------------------------------------------------------
    print("=== Loaded SeMRA Prep Config ===")
    print(f"Config path          : {yaml_path}")
    print(f"Data directory       : {cfg.data_dir}")
    print(f"Database path        : {cfg.db_path}")
    print(f"Header cache         : {cfg.header_cache}")
    print(f"Log file             : {cfg.log_file}")
    print(f"Landscapes           : {cfg.landscapes}")
    print(f"Pipeline flags       : {cfg.pipeline}")
    print("================================")

    return cfg


# ============================================================
# UNIT TEST
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
