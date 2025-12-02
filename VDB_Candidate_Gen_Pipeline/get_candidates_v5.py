import argparse
import pandas as pd
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch


def embed(text, tokenizer, model):
    """Encode text using SapBERT."""
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        output = model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()


def load_metadata(csv_path):
    """
    Load ontology CSV with:
    curie, label, synonyms, definition
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Synonyms/definition should be strings
    df["synonyms"] = df.get("synonyms", "").fillna("")
    df["definition"] = df.get("definition", "").fillna("")

    return df.set_index("curie")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--target_collection", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--metadata", default="data/mp.csv", help="Ontology CSV")
    parser.add_argument("--save_tsv", default=None, help="TSV output file")
    args = parser.parse_args()

    # Load SapBERT or finetuned model
    model_name = (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        if args.model == "sapbert"
        else args.model
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Embed the query label
    query_vec = embed(args.label, tokenizer, model)

    # Load metadata (for synonyms + definitions)
    meta = load_metadata(args.metadata)

    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")

    # Vector search
    results = client.query_points(
        collection_name=args.target_collection,
        query=query_vec,
        limit=args.topk,
        with_payload=True,
        with_vectors=False,
    ).points

    print("\nTop candidates:\n")

    rows = []

    for i, sp in enumerate(results, 1):
        payload = sp.payload
        score = sp.score

        curie = payload.get("curie", "")
        label = payload.get("label", "")

        synonyms = ""
        definition = ""

        if curie in meta.index:
            synonyms = meta.loc[curie].get("synonyms", "")
            definition = meta.loc[curie].get("definition", "")

        # --- Print nicely ---
        print(f"{i}. {curie} | {label} (score={score:.4f})")
        print(f"   Synonyms: {synonyms}")
        print(f"   Definition: {definition}\n")

        # --- Add for TSV ---
        rows.append(
            {
                "rank": i,
                "curie": curie,
                "label": label,
                "score": score,
                "synonyms": synonyms,
                "definition": definition,
            }
        )

    # --- Save TSV if requested ---
    if args.save_tsv:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.save_tsv, sep="\t", index=False)
        print(f"\n✔ TSV saved → {args.save_tsv}")


if __name__ == "__main__":
    main()
