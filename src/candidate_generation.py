from typing import List, Dict
import itertools

def lexical_candidates(src_id: str, all_targets: List[str], top_k: int=25) -> List[str]:
    # very naive: rank by shared prefix token; replace with real lexical ranking
    ranked = sorted(all_targets, key=lambda t: int(t.split(':')[0] == src_id.split(':')[0]), reverse=True)
    return ranked[:top_k]

def embed_candidates(src_id: str, all_targets: List[str], top_k: int=25) -> List[str]:
    # placeholder: alternate targets for diversity
    return all_targets[:top_k]

def interleave(a: List[str], b: List[str], k: int) -> List[str]:
    merged = [x for pair in itertools.zip_longest(a, b) for x in pair if x]
    seen, out = set(), []
    for c in merged:
        if c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= k:
            break
    return out

def generate_candidates(src_id: str, all_targets: List[str], lexical_top_k=25, embed_top_k=25) -> List[str]:
    lex = lexical_candidates(src_id, all_targets, lexical_top_k)
    emb = embed_candidates(src_id, all_targets, embed_top_k)
    return interleave(lex, emb, k=max(lexical_top_k, embed_top_k))
