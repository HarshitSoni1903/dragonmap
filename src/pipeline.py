# src/pipeline.py
import os
import sys
import yaml
import logging
from typing import Dict, List, Tuple

# Local modules
from src.semra_adapter import load_semra_priors_from_sssom, PriorConfig
from src.candidate_generation import generate_candidates
from src.bertmap_integration import score_pairs
from src.fusion import fuse_scores
from src.thresholding import select
from src.validation import evaluate_at_1

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pipeline")


def read_config(cfg_path: str) -> dict:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_semra_priors(cfg: dict) -> Dict[Tuple[str, str], float]:
    """Load SeMRA mappings from SSSOM and compute provenance-weighted priors."""
    semra_cfg = cfg.get("semra", {})
    mode = semra_cfg.get("mode", "sssom_file")

    if mode != "sssom_file":
        # You can extend this later to support multi_landscape or other modes.
        logger.warning(
            "Only mode 'sssom_file' is implemented in this pipeline. Got '%s'. Falling back to sssom_file.",
            mode,
        )

    sssom_path = semra_cfg.get("sssom_path")
    if not sssom_path or not os.path.exists(sssom_path):
        raise FileNotFoundError(
            f"SeMRA SSSOM file not found. Update config semra.sssom_path: {sssom_path}"
        )

    pcfg = PriorConfig(
        allowed_predicates=semra_cfg.get("allowed_predicates", []),
        min_confidence=semra_cfg.get("min_confidence", 0.5),
        evidence_agg=semra_cfg.get("evidence_agg", "max"),
        hop_decay=semra_cfg.get("hop_decay", 0.95),
    )

    logger.info("Loading SeMRA SSSOM from %s", sssom_path)
    priors = load_semra_priors_from_sssom(sssom_path, pcfg)
    logger.info("Loaded %d SeMRA prior edges", len(priors))
    return priors


def ids_from_priors(
    semra_priors: Dict[Tuple[str, str], float],
) -> Tuple[List[str], List[str]]:
    """Derive source and target ID sets from the prior dictionary keys."""
    src_ids = sorted({s for (s, _) in semra_priors.keys()})
    all_targets = sorted({t for (_, t) in semra_priors.keys()})
    return src_ids, all_targets


def main(cfg_path: str = "config.yaml") -> None:
    # 0) Read config
    cfg = read_config(cfg_path)
    logger.info("Experiment: %s", cfg.get("experiment_name", "unnamed_experiment"))

    # 1) SeMRA priors (from SSSOM)
    semra_priors = build_semra_priors(cfg)

    # 2) Build ID lists from priors
    src_ids, all_targets = ids_from_priors(semra_priors)
    if not src_ids or not all_targets:
        logger.warning(
            "No source or target IDs found from SeMRA priors. Nothing to do."
        )
        return

    # 3) Pipeline settings
    retrieval_cfg = cfg.get("retrieval", {})
    fusion_alpha = cfg.get("fusion", {}).get("alpha", 0.7)
    thr_cfg = cfg.get("thresholding", {})
    gold_path = cfg.get("paths", {}).get("gold_set")

    if not gold_path or not os.path.exists(gold_path):
        logger.warning(
            "Gold file not found at %s. Evaluation will be skipped.",
            gold_path,
        )

    predictions: List[Tuple[str, str]] = []

    # 4) Iterate over sources
    for i, src in enumerate(src_ids, 1):
        if i % 100 == 0:
            logger.info("Scoring %d / %d sources...", i, len(src_ids))

        # 4a) Candidate generation (lexical + embedding interleaving)
        candidates = generate_candidates(
            src,
            all_targets,
            lexical_top_k=retrieval_cfg.get("lexical_top_k", 25),
            embed_top_k=retrieval_cfg.get("embed_top_k", 25),
        )
        if not candidates:
            continue

        # 4b) (Mock) BERTMap scores for (src, candidate) pairs
        b_scores = score_pairs(src, candidates)

        # 4c) SeMRA priors lookup for these candidates
        priors_for_src = {
            t: semra_priors.get((src.upper(), t.upper()), 0.0) for t in candidates
        }

        # 4d) Fuse: s* = alpha * s_b + (1 - alpha) * p_se
        fused = fuse_scores(b_scores, priors_for_src, alpha=fusion_alpha)

        # 4e) Thresholding / tie handling
        chosen = select(
            fused,
            top_k=thr_cfg.get("top_k", 1),
            prob_min=thr_cfg.get("prob_min", 0.6),
            tie_margin=thr_cfg.get("tie_margin", 0.02),
        )
        if chosen:
            best_tgt = max(chosen, key=lambda kv: kv[1])[0]
            predictions.append((src, best_tgt))

    logger.info("Total predictions: %d", len(predictions))

    # 5) Evaluate (optional)
    if gold_path and os.path.exists(gold_path):
        metrics = evaluate_at_1(predictions, gold_path)
        logger.info("Evaluation metrics: %s", metrics)
        print("Predictions:", predictions)
        print("Metrics:", metrics)
    else:
        print("Predictions:", predictions)
        logger.info("No gold file provided. Skipped evaluation.")


if __name__ == "__main__":
    # Allow: python -m src.pipeline [config_path]
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg_arg)
