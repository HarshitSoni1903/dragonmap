from typing import Dict

def fuse_scores(bert_scores: Dict[str, float], semra_priors: Dict[str, float], alpha: float=0.7) -> Dict[str, float]:
    fused = {}
    for tgt, s_b in bert_scores.items():
        p_se = semra_priors.get(tgt, 0.0)
        s_star = alpha * s_b + (1 - alpha) * p_se
        fused[tgt] = max(0.0, min(1.0, s_star))
    return fused
