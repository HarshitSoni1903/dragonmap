import os
import math
import yaml
import torch
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from prep_eval import run_full_evaluation



# ============================
#  DATASET
# ============================

class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text1 = row["subject_label"]
        text2 = row["object_label"]

        enc1 = self.tokenizer(
            text1,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids_1": enc1["input_ids"].squeeze(0),
            "attention_mask_1": enc1["attention_mask"].squeeze(0),
            "input_ids_2": enc2["input_ids"].squeeze(0),
            "attention_mask_2": enc2["attention_mask"].squeeze(0),
        }


# ============================
#  MULTI-SIMILARITY LOSS
# ============================

def multi_similarity_loss(embeddings_1, embeddings_2, alpha, beta, margin):
    """
    embeddings_1 and embeddings_2: (batch_size, hidden_dim)
    """
    batch_size = embeddings_1.size(0)

    # Normalize embeddings
    e1 = torch.nn.functional.normalize(embeddings_1, p=2, dim=1)
    e2 = torch.nn.functional.normalize(embeddings_2, p=2, dim=1)

    # Positive similarities (paired)
    pos_sim = torch.sum(e1 * e2, dim=1)  # shape: [batch]

    # All pairwise similarities (in-batch)
    all_sims = torch.matmul(e1, e2.t())  # shape: [batch, batch]

    # Mask out diagonal for negatives
    neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=all_sims.device)
    neg_sims = all_sims[neg_mask].view(batch_size, batch_size - 1)

    # Multi-Similarity Loss
    pos_exp = torch.exp(-alpha * (pos_sim - margin))
    neg_exp = torch.exp(beta * (neg_sims - margin))

    pos_term = (1.0 / alpha) * torch.log1p(pos_exp).mean()
    neg_term = (1.0 / beta) * torch.log1p(neg_exp.sum(dim=1)).mean()

    return pos_term + neg_term


# ============================
#  UTILS
# ============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_output_dir(cfg):
    exp_name = cfg["experiment"]["name"]
    base_dir = cfg["experiment"]["output_dir"]
    out_dir = os.path.join(base_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_or_download_model(model_name: str, model_dir: str):
    """
    If model_dir exists and contains a HuggingFace model (config.json),
    load it locally.
    Otherwise download from HuggingFace and save to model_dir.
    Returns (model, tokenizer).
    """
    model_dir = os.path.expanduser(model_dir)
    config_path = os.path.join(model_dir, "config.json")

    # ----------------------------------------
    # 1. Try to load from disk
    # ----------------------------------------
    if os.path.isfile(config_path):
        print(f"[MODEL] Found local model at: {model_dir}")
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer

    # ----------------------------------------
    # 2. Download from HF
    # ----------------------------------------
    print(f"[MODEL] Local model not found at {model_dir}. Downloading {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"[MODEL] Downloaded and saved to: {model_dir}")
    return model, tokenizer



# ============================
#  TRAINING FUNCTION
# ============================

def train(config_path: str, resume: bool = False):
    cfg = load_config(config_path)
    out_dir = build_output_dir(cfg)

    # ---- Save a copy of config for reproducibility ----
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    model_name = cfg["model"]["name_or_path"]
    train_csv = cfg["data"]["train_csv"]
    val_csv   = cfg["data"]["val_csv"]

    tcfg = cfg["training"]
    lcfg = cfg["loss"]

    # --- robust safe parsing ---
    lr = float(tcfg["lr"])
    weight_decay = float(tcfg["weight_decay"])
    eps = float(tcfg["eps"])
    warmup_ratio = float(tcfg["warmup_ratio"])
    max_grad_norm = float(tcfg["max_grad_norm"])

    batch_size = int(tcfg["batch_size"])
    max_seq_len = int(tcfg["max_seq_length"])
    epochs = int(tcfg["epochs"])
    num_workers = int(tcfg["num_workers"])
    gradient_accum_steps = int(tcfg["gradient_accumulation_steps"])

    betas = tuple(float(b) for b in tcfg["betas"])
    fp16 = bool(tcfg["fp16"])
    seed = int(tcfg["seed"])
    log_every = int(tcfg["log_every"])


    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    model_dir = cfg["model"].get("local_dir", model_name)
    model, tokenizer = load_or_download_model(model_name, model_dir)
    model.to(device)

    train_dataset = PairDataset(train_df, tokenizer, max_seq_len)
    val_dataset   = PairDataset(val_df,   tokenizer, max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        betas=betas,
    )

    # number of optimizer steps (not batches) when using grad_accumulation
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accum_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    # ---- Checkpoint paths ----
    last_ckpt_path = os.path.join(out_dir, "checkpoint_last.pt")
    best_ckpt_path = os.path.join(out_dir, "checkpoint_best.pt")

    start_epoch = 0
    best_val_loss = float("inf")

    # ---- Resume if requested ----
    if resume and os.path.isfile(last_ckpt_path):
        print(f"[INFO] Resuming from checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
    elif resume:
        print(f"[WARN] --resume was set but checkpoint not found at {last_ckpt_path}")

    # ============================
    #  TRAINING LOOP
    # ============================

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader):
            input_ids_1 = batch["input_ids_1"].to(device, non_blocking=True)
            attn_1 = batch["attention_mask_1"].to(device, non_blocking=True)
            input_ids_2 = batch["input_ids_2"].to(device, non_blocking=True)
            attn_2 = batch["attention_mask_2"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=fp16):
                out1 = model(input_ids=input_ids_1, attention_mask=attn_1).last_hidden_state[:, 0, :]
                out2 = model(input_ids=input_ids_2, attention_mask=attn_2).last_hidden_state[:, 0, :]

                loss = multi_similarity_loss(
                    out1, out2,
                    lcfg["alpha"],
                    lcfg["beta"],
                    lcfg["margin"],
                )

                # gradient accumulation
                loss = loss / gradient_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * gradient_accum_steps  # unscaled for logging

            if (step + 1) % gradient_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            if (step + 1) % log_every == 0:
                avg_so_far = total_loss / (step + 1)
                print(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{len(train_loader)} "
                      f"Loss: {avg_so_far:.4f}",
                      flush=True)

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}", flush=True)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids_1 = batch["input_ids_1"].to(device, non_blocking=True)
                attn_1 = batch["attention_mask_1"].to(device, non_blocking=True)
                input_ids_2 = batch["input_ids_2"].to(device, non_blocking=True)
                attn_2 = batch["attention_mask_2"].to(device, non_blocking=True)

                out1 = model(input_ids=input_ids_1, attention_mask=attn_1).last_hidden_state[:, 0, :]
                out2 = model(input_ids=input_ids_2, attention_mask=attn_2).last_hidden_state[:, 0, :]

                batch_loss = multi_similarity_loss(
                    out1, out2,
                    lcfg["alpha"],
                    lcfg["beta"],
                    lcfg["margin"],
                )
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Val Loss: {avg_val_loss:.4f}", flush=True)

        # ---- Save checkpoints ----
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
        }

        torch.save(ckpt_state, last_ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_state, best_ckpt_path)
            print(f"[Epoch {epoch+1}] New best checkpoint saved (val_loss={best_val_loss:.4f})",
                  flush=True)

    # ---- Save final model (weights only) ----
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[DONE] Saved finetuned SapBERT + tokenizer to: {out_dir}", flush=True)
    
    return model, tokenizer, cfg, device
    
    
def evaluate_on_test(model, tokenizer, cfg, device):
    test_csv = cfg["data"]["test_csv"]
    test_df = pd.read_csv(test_csv)

    model.eval()
    sims = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            t1, t2 = row["subject_label"], row["object_label"]

            enc1 = tokenizer(
                t1, max_length=cfg["training"]["max_seq_length"],
                truncation=True, padding="max_length", return_tensors="pt"
            ).to(device)

            enc2 = tokenizer(
                t2, max_length=cfg["training"]["max_seq_length"],
                truncation=True, padding="max_length", return_tensors="pt"
            ).to(device)

            e1 = model(**enc1).last_hidden_state[:,0,:]
            e2 = model(**enc2).last_hidden_state[:,0,:]

            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)

            sim = torch.sum(e1 * e2, dim=1).item()
            sims.append(sim)

    print("\n===== Test Evaluation =====")
    print(f"Mean cosine similarity: {np.mean(sims):.4f}")
    print(f"Median cosine similarity: {np.median(sims):.4f}")
    print(f"Min: {np.min(sims):.4f} | Max: {np.max(sims):.4f}")
    print("==========================\n")
    run_full_evaluation(model, tokenizer, cfg, device, sims, test_df, prefix="finetuned")


def evaluate_base_model_on_test(cfg, device):
    """
    Loads the *original* HF base model (without fine-tuning)
    and computes its cosine similarity distribution on the same test set.
    """

    base_name = cfg["model"]["name_or_path"]      # e.g. "bert-base-uncased" or HF repo
    test_csv = cfg["data"]["test_csv"]
    test_df = pd.read_csv(test_csv)

    print("\n===== Loading BASE model for comparison =====")
    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModel.from_pretrained(base_name).to(device)
    base_model.eval()

    sims = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            t1, t2 = row["subject_label"], row["object_label"]

            enc1 = base_tokenizer(
                t1, max_length=cfg["training"]["max_seq_length"],
                truncation=True, padding="max_length", return_tensors="pt"
            ).to(device)

            enc2 = base_tokenizer(
                t2, max_length=cfg["training"]["max_seq_length"],
                truncation=True, padding="max_length", return_tensors="pt"
            ).to(device)

            e1 = base_model(**enc1).last_hidden_state[:, 0, :]
            e2 = base_model(**enc2).last_hidden_state[:, 0, :]

            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)

            sim = torch.sum(e1 * e2, dim=1).item()
            sims.append(sim)

    print("\n===== Base Model Test Evaluation =====")
    print(f"Mean cosine similarity: {np.mean(sims):.4f}")
    print(f"Median cosine similarity: {np.median(sims):.4f}")
    print(f"Min: {np.min(sims):.4f} | Max: {np.max(sims):.4f}")
    print("==========================\n")
    run_full_evaluation(base_model, base_tokenizer, cfg, device, sims, test_df, prefix="base")



# ============================
#  ENTRY POINT
# ============================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if available")
    args = parser.parse_args()

    model, tokenizer, cfg, device = train(args.config, resume=args.resume)
    evaluate_on_test(model, tokenizer, cfg, device)
    
    evaluate_base_model_on_test(cfg, device)
