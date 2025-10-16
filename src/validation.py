import pandas as pd
from typing import List, Tuple

def evaluate_at_1(preds: List[Tuple[str, str]], gold_csv_path: str) -> dict:
    gold = pd.read_csv(gold_csv_path)
    gold_pairs = set(zip(gold['src_id'], gold['tgt_id']))
    correct = sum((p in gold_pairs) for p in preds)
    n = len(preds) if preds else 1
    precision_at_1 = correct / n
    accuracy = precision_at_1  # same when one prediction per source
    return {'precision_at_1': precision_at_1, 'accuracy': accuracy}
