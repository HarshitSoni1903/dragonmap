import argparse
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--target_collection", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--topk", type=int, default=200)
    args = parser.parse_args()

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    # Encode query text
    text = args.label
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    # --- Connect to Qdrant ---
    client = QdrantClient(url="http://localhost:6333")

    # --- Search top K ---
    results = client.query_points(
        collection_name=args.target_collection,
        query=emb.tolist(),
        limit=args.topk,
        with_payload=True,
        with_vectors=False,
    )

    print("\nTop candidates:")
    for i, r in enumerate(results.points, 1):
        label = r.payload.get("label", "")
        cid = r.payload.get("id", "")
        score = r.score
        print(f"{i}. {cid} | {label} | (score={score:.4f})")


if __name__ == "__main__":
    main()
