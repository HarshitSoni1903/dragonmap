# Thin wrapper placeholder for BERTMap scoring.
# Replace with actual BERTMap integration (loading model, encoding pairs, etc.)
from typing import List, Dict
import random

def score_pairs(src_id: str, candidates: List[str]) -> Dict[str, float]:
    # return pseudo-probabilities in [0.5, 0.95] for demo purposes
    rng = random.Random(hash(src_id) % (2**32))
    return {c: 0.5 + 0.45 * rng.random() for c in candidates}
