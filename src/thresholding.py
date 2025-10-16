from typing import Dict, List, Tuple

def select(fused: Dict[str, float], top_k: int=1, prob_min: float=0.6, tie_margin: float=0.02) -> List[Tuple[str, float]]:
    items = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return []
    top = [items[0]]
    # include near-ties within margin
    for tgt, p in items[1:]:
        if len(top) >= top_k and p < top[0][1] - tie_margin:
            break
        if p >= prob_min:
            top.append((tgt, p))
    return top[:max(top_k, len(top))]
