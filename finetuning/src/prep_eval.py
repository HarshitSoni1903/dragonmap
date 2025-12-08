# prep_eval.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ============================================================
# Helper: extract embeddings for TSNE / UMAP
# ============================================================
def extract_embeddings(model, tokenizer, df, device, max_len, n_limit=10000):
    """
    Returns an array of concatenated embeddings [e1||e2]
    using CLS-token for each text pair.
    """
    if len(df) > n_limit:
        df = df.sample(n_limit, random_state=42).reset_index(drop=True)

    embeds = []

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            t1, t2 = row["subject_label"], row["object_label"]

            enc1 = tokenizer(
                t1,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            enc2 = tokenizer(
                t2,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            out1 = model(**enc1).last_hidden_state[:, 0, :]
            out2 = model(**enc2).last_hidden_state[:, 0, :]

            n1 = torch.nn.functional.normalize(out1, p=2, dim=1)[0].cpu().numpy()
            n2 = torch.nn.functional.normalize(out2, p=2, dim=1)[0].cpu().numpy()

            embeds.append(np.concatenate([n1, n2]))

    return np.array(embeds)


# ============================================================
# Helper: Save histogram of cos sims
# ============================================================
def plot_similarity_histogram(sims, out_dir, prefix):
    plt.figure(figsize=(6,4))
    plt.hist(sims, bins=40, color="blue", alpha=0.7)
    plt.title(f"Cosine Similarity Distribution ({prefix})")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_similarity_hist.png")
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# Helper: TSNE + UMAP visualization
# ============================================================
def run_tsne_umap(embeddings, out_dir, prefix):
    # Standardize for stability
    embeddings = StandardScaler().fit_transform(embeddings)

    # ---------------------------
    # TSNE
    # ---------------------------
    print(f"[prep_eval] Running TSNE for {prefix}...")
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30)
    tsne_out = tsne.fit_transform(embeddings)

    plt.figure(figsize=(6,6))
    plt.scatter(tsne_out[:,0], tsne_out[:,1], s=3, alpha=0.5)
    plt.title(f"TSNE plot ({prefix})")
    plt.tight_layout()
    tsne_path = os.path.join(out_dir, f"{prefix}_tsne.png")
    plt.savefig(tsne_path)
    plt.close()

    # ---------------------------
    # UMAP
    # ---------------------------
    if HAS_UMAP:
        print(f"[prep_eval] Running UMAP for {prefix}...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_out = reducer.fit_transform(embeddings)

        plt.figure(figsize=(6,6))
        plt.scatter(umap_out[:,0], umap_out[:,1], s=3, alpha=0.5, color="green")
        plt.title(f"UMAP plot ({prefix})")
        plt.tight_layout()
        umap_path = os.path.join(out_dir, f"{prefix}_umap.png")
        plt.savefig(umap_path)
        plt.close()
    else:
        umap_path = None

    return tsne_path, umap_path


# ============================================================
# MAIN ENTRY FUNCTION
# ============================================================
def run_full_evaluation(model, tokenizer, cfg, device, sims, test_df, prefix="finetuned"):
    """
    Called inside train.py evaluation functions.
    Performs histogram + TSNE/UMAP and writes results to experiment folder.
    """
    out_dir = os.path.join(
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["name"]
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[prep_eval] Running full evaluation for {prefix}...")

    # ---------------------------------------------------------
    # 1. Save histogram
    # ---------------------------------------------------------
    hist_path = plot_similarity_histogram(sims, out_dir, prefix)
    print(f"[prep_eval] Histogram saved to: {hist_path}")

    # ---------------------------------------------------------
    # 2. Extract embeddings for TSNE/UMAP
    # ---------------------------------------------------------
    max_len = cfg["training"]["max_seq_length"]
    embeddings = extract_embeddings(model, tokenizer, test_df, device, max_len)

    # ---------------------------------------------------------
    # 3. TSNE + UMAP
    # ---------------------------------------------------------
    tsne_path, umap_path = run_tsne_umap(embeddings, out_dir, prefix)

    print(f"[prep_eval] TSNE saved to: {tsne_path}")
    if umap_path:
        print(f"[prep_eval] UMAP saved to: {umap_path}")
    else:
        print("[prep_eval] UMAP not available (install umap-learn).")

    print(f"[prep_eval] Evaluation finished for {prefix}.")
