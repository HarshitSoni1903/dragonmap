from typing import Dict, Iterable, Tuple
from dataclasses import dataclass
import semra, semra.io
from semra.struct import SimpleEvidence, ReasonedEvidence, Mapping


@dataclass
class PriorConfig:
    allowed_predicates: Iterable[str]
    min_confidence: float = 0.5
    evidence_agg: str = "max"  # "max" | "mean" | "weighted"
    hop_decay: float = 0.95


def _evidence_confidence(ev, hop_decay: float) -> float:
    if isinstance(ev, SimpleEvidence):
        return float(getattr(ev, "confidence", 0.6) or 0.6)
    if isinstance(ev, ReasonedEvidence):
        inner = ev.mappings or []
        scores = []
        for m in inner:
            scores.extend(
                _evidence_confidence(e2, hop_decay) for e2 in getattr(m, "evidence", [])
            )
        base = max(scores) if scores else 0.6
        hops = 2
        return base * (hop_decay ** (hops - 1))
    return 0.6


def prior_from_mapping(m: Mapping, cfg: PriorConfig) -> float:
    predicate = getattr(m.predicate, "curie", None) or getattr(m.predicate, "name", "")
    if cfg.allowed_predicates and predicate not in cfg.allowed_predicates:
        return 0.0
    scores = [_evidence_confidence(ev, cfg.hop_decay) for ev in (m.evidence or [])]
    if not scores:
        return 0.0
    score = (sum(scores) / len(scores)) if cfg.evidence_agg == "mean" else max(scores)
    return score if score >= cfg.min_confidence else 0.0


def load_semra_priors_from_sssom(
    path: str, cfg: PriorConfig
) -> Dict[Tuple[str, str], float]:
    mappings = semra.io.from_sssom(path)
    priors = {}
    for m in mappings:
        s = f"{m.subject.prefix}:{m.subject.identifier}".upper()
        t = f"{m.object.prefix}:{m.object.identifier}".upper()
        p = prior_from_mapping(m, cfg)
        if p > 0:
            priors[(s, t)] = max(priors.get((s, t), 0.0), p)
    return priors
