import csv
from typing import Dict, Tuple, List
import math

class SeMRAGraph:
    def __init__(self, prefix_map: Dict[str, str], hop_decay: float, priors: Dict[str, float]):
        self.prefix_map = prefix_map
        self.hop_decay = hop_decay
        self.priors = priors
        # store edges as ((s,t) -> best_prior)
        self.edge_prior = {}

    @staticmethod
    def _normalize_id(curie: str, prefix_map: Dict[str, str], normalize_case: bool=True, strip_version: bool=True) -> str:
        # Simple CURIE normalizer; extend as needed
        curie = curie.strip()
        if normalize_case:
            curie = curie.upper()
        if strip_version and "|" in curie:
            curie = curie.split("|")[0]
        return curie

    def load_tsv(self, path: str, normalize_case=True, strip_version=True):
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                s = self._normalize_id(row['subject'], self.prefix_map, normalize_case, strip_version)
                t = self._normalize_id(row['object'], self.prefix_map, normalize_case, strip_version)
                ptype = row.get('provenance_type', 'inferred').lower()
                hops = int(row.get('hops', 1) or 1)
                base = self.priors.get(ptype, 0.6)
                # attenuate by hops: base * hop_decay^(hops-1)
                prior = base * (self.hop_decay ** max(0, hops-1))
                key = (s, t)
                self.edge_prior[key] = max(self.edge_prior.get(key, 0.0), min(prior, 1.0))

    def prior(self, src_id: str, tgt_id: str) -> float:
        return self.edge_prior.get((src_id, tgt_id), 0.0)
