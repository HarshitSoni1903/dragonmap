# import argparse
# import pandas as pd
# from qdrant_client import QdrantClient
# from transformers import AutoTokenizer, AutoModel
# import torch
# import os


# def embed(text, tokenizer, model):
#     """Encode text using SapBERT."""
#     with torch.no_grad():
#         tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
#         output = model(**tokens)
#         return output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()


# def load_metadata(csv_path):
#     """
#     Load ontology CSV with:
#     curie, label, synonyms, definition
#     """
#     df = pd.read_csv(csv_path)

#     # Normalize column names
#     df.columns = [c.lower() for c in df.columns]

#     # Synonyms/definition should be strings
#     df["synonyms"] = df.get("synonyms", "").fillna("")
#     df["definition"] = df.get("definition", "").fillna("")

#     return df.set_index("curie")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--label", required=True)
#     parser.add_argument("--target_collection", required=True)
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--topk", type=int, default=20)
#     parser.add_argument("--metadata", default="data/mp.csv", help="Ontology CSV")
#     parser.add_argument(
#         "--out_csv",
#         default=None,
#         help="Optional: path to CSV file to save candidates",
#     )
#     args = parser.parse_args()

#     # Load SapBERT or finetuned model
#     model_name = (
#         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#         if args.model == "sapbert"
#         else args.model
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     model.eval()

#     # Embed the query label
#     query_vec = embed(args.label, tokenizer, model)

#     # Load metadata (for synonyms + definitions)
#     meta = load_metadata(args.metadata)

#     # Connect to Qdrant
#     client = QdrantClient(url="http://localhost:6333")

#     # Vector search (THIS is the behavior that already works for you)
#     results = client.query_points(
#         collection_name=args.target_collection,
#         query=query_vec,
#         limit=args.topk,
#         with_payload=True,
#         with_vectors=False,
#     ).points

#     print("\nTop candidates:\n")

#     rows = []

#     for i, sp in enumerate(results, 1):
#         payload = sp.payload
#         score = sp.score

#         curie = payload.get("curie", "")
#         label = payload.get("label", "")

#         synonyms = ""
#         definition = ""

#         if curie in meta.index:
#             synonyms = meta.loc[curie].get("synonyms", "")
#             definition = meta.loc[curie].get("definition", "")

#         # --- Print nicely (same as before) ---
#         print(f"{i}. {curie} | {label} (score={score:.4f})")
#         print(f"   Synonyms: {synonyms}")
#         print(f"   Definition: {definition}\n")

#         # --- Add for CSV ---
#         rows.append(
#             {
#                 "rank": i,
#                 "curie": curie,
#                 "label": label,
#                 "score": score,
#                 "synonyms": synonyms,
#                 "definition": definition,
#             }
#         )

#     # --- Save CSV if requested ---
#     if args.out_csv:
#         os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
#         out_df = pd.DataFrame(rows)
#         out_df.to_csv(args.out_csv, index=False)
#         print(f"\n✔ CSV saved → {args.out_csv}")


# if __name__ == "__main__":
#     main()
import argparse
import pandas as pd
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
import os
from datetime import datetime
import re


def clean_label_for_filename(label: str):
    """Convert label to safe filename."""
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return label.strip("_")


def embed(text, tokenizer, model):
    """Encode text using SapBERT."""
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        output = model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()


def load_metadata(csv_path):
    """Load ontology CSV with curie, label, synonyms, definition."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df["synonyms"] = df.get("synonyms", "").fillna("")
    df["definition"] = df.get("definition", "").fillna("")
    return df.set_index("curie")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--target_collection", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--metadata", default="data/mp.csv")
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional explicit CSV path. If not set → auto-naming is used.",
    )
    parser.add_argument(
        "--timestamp", action="store_true", help="Append timestamp to filename"
    )
    args = parser.parse_args()

    # ----------------------
    # Load embedding model
    # ----------------------
    model_name = (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        if args.model == "sapbert"
        else args.model
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # ----------------------
    # Embed query
    # ----------------------
    query_vec = embed(args.label, tokenizer, model)

    # ----------------------
    # Metadata load
    # ----------------------
    meta = load_metadata(args.metadata)

    # ----------------------
    # Qdrant query
    # ----------------------
    client = QdrantClient(url="http://localhost:6333")
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

        synonyms = meta.loc[curie].get("synonyms", "") if curie in meta.index else ""
        definition = (
            meta.loc[curie].get("definition", "") if curie in meta.index else ""
        )

        print(f"{i}. {curie} | {label} (score={score:.4f})")
        print(f"   Synonyms: {synonyms}")
        print(f"   Definition: {definition}\n")

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

    # ----------------------
    # Auto-generate CSV filename if not provided
    # ----------------------
    if args.out_csv is None:
        os.makedirs("candidate_outputs", exist_ok=True)

        label_clean = clean_label_for_filename(args.label)

        fname = f"{label_clean}__{args.target_collection}__{args.model}__top{args.topk}"

        if args.timestamp:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname += f"__{stamp}"

        fname += ".csv"
        out_csv = os.path.join("candidate_outputs", fname)
    else:
        out_csv = args.out_csv

    # ----------------------
    # Save CSV
    # ----------------------
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\n✔ CSV saved → {out_csv}")


if __name__ == "__main__":
    main()
