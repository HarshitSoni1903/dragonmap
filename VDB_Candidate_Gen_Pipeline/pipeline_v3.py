import argparse
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from transformers import AutoTokenizer, AutoModel
import torch

#############################################
#               CONFIG
#############################################

QDRANT_URL = "http://localhost:6333"
EMBED_DIR = "embeddings"

os.makedirs(EMBED_DIR, exist_ok=True)

#############################################
#           MODEL LOADING
#############################################


def load_model(model_name):
    print(f"Loading SapBERT: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


#############################################
#           EMBEDDING FUNCTION
#############################################


def embed(text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        outputs = model(**inputs)
        # mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding.tolist()


#############################################
#           FILE LOADING
#############################################


def load_owl(path):
    # Convert OWL → DataFrame
    print(f"Loading OWL: {path}")
    from owlready2 import get_ontology

    onto = get_ontology(path).load()
    rows = []

    for cls in onto.classes():
        label = cls.label[0] if cls.label else ""
        definition = cls.comment[0] if cls.comment else ""
        synonyms = "; ".join(cls.altLabel) if hasattr(cls, "altLabel") else ""

        rows.append(
            {
                "curie": cls.name,
                "label": label,
                "definition": definition,
                "synonyms": synonyms,
            }
        )
    return pd.DataFrame(rows)


def load_csv(path):
    print(f"Loading CSV: {path}")
    return pd.read_csv(path)


def load_input(path):
    if path.endswith(".owl"):
        return load_owl(path)
    elif path.endswith(".csv"):
        return load_csv(path)
    else:
        raise ValueError("Unsupported input format: must be .owl or .csv")


#############################################
#            CACHING UTILITIES
#############################################


def get_cache_paths(collection):
    emb_path = f"{EMBED_DIR}/{collection}.npy"
    payload_path = f"{EMBED_DIR}/{collection}_payloads.pkl"
    return emb_path, payload_path


def save_cache(collection, vectors, payloads):
    emb_path, payload_path = get_cache_paths(collection)

    np.save(emb_path, np.array(vectors))
    with open(payload_path, "wb") as f:
        pickle.dump(payloads, f)

    print(f"✔ Saved cache → {emb_path}")
    print(f"✔ Saved payloads → {payload_path}")


def load_cache(collection):
    emb_path, payload_path = get_cache_paths(collection)

    if os.path.exists(emb_path) and os.path.exists(payload_path):
        print(f"⚡ Cache found for {collection} — loading...")
        vectors = np.load(emb_path)
        with open(payload_path, "rb") as f:
            payloads = pickle.load(f)
        print(f"✔ Loaded {len(vectors)} cached vectors")
        return vectors, payloads

    return None, None


#############################################
#      VECTOR GENERATION WITH CACHING
#############################################


def generate_or_load_embeddings(df, tokenizer, model, collection):
    # Try loading cache
    cached_vecs, cached_payloads = load_cache(collection)
    if cached_vecs is not None:
        print("⚡ Using cached embeddings — skipping embedding computation")
        return cached_vecs, cached_payloads

    # Else compute embeddings
    print(f"Generating embeddings for {collection} ({len(df)} concepts)")
    vectors = []
    payloads = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = f"{row['label']} {row['synonyms']} {row['definition']}"
        vec = embed(text, tokenizer, model)

        vectors.append(vec)
        payloads.append(
            {
                "curie": row["curie"],
                "label": row["label"],
            }
        )

    # Save cache
    save_cache(collection, vectors, payloads)

    return np.array(vectors), payloads


#############################################
#              QDRANT UPLOAD
#############################################


def create_collection(client, name, dim):
    print(f"Creating collection: {name}")
    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upload_vectors(client, collection, vectors, payloads, batch_size=512):
    print(f"Uploading {len(vectors)} vectors → {collection}")

    for start in tqdm(range(0, len(vectors), batch_size)):
        end = start + batch_size

        points = [
            PointStruct(
                id=int(i),
                vector=vectors[i].tolist()
                if isinstance(vectors, np.ndarray)
                else vectors[i],
                payload=payloads[i],
            )
            for i in range(start, min(end, len(vectors)))
        ]

        client.upsert(collection_name=collection, points=points)

    print(f"✔ Upload complete: {collection}")


#############################################
#                  MAIN
#############################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to OWL or CSV file")
    parser.add_argument(
        "--model", required=True, help="sapbert OR path to finetuned model"
    )
    args = parser.parse_args()

    # Choose model
    if args.model == "sapbert":
        model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    else:
        model_name = args.model  # local finetuned path

    # Determine collection name
    base = os.path.basename(args.data).split(".")[0]
    model_tag = "sapbert" if args.model == "sapbert" else "sapbert_ft"
    collection_name = f"{base}_{model_tag}"

    # Load model
    tokenizer, model = load_model(model_name)

    # Load input file
    df = load_input(args.data)

    # Init Qdrant
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    # Embedding step (cached)
    vectors, payloads = generate_or_load_embeddings(
        df, tokenizer, model, collection_name
    )

    # Create collection
    create_collection(client, collection_name, dim=len(vectors[0]))

    # Upload batched
    upload_vectors(client, collection_name, vectors, payloads)


if __name__ == "__main__":
    main()
