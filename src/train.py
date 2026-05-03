"""
train.py
--------
Training loop for the Japanese–English Transformer model.

Typical usage
~~~~~~~~~~~~~
.. code-block:: bash

    python src/train.py \
        --train_src  data/train.ja \
        --train_tgt  data/train.en \
        --val_src    data/val.ja \
        --val_tgt    data/val.en \
        --sp_model   data/spm.model \
        --checkpoint_dir checkpoints/ \
        --epochs 20 \
        --batch_size 64 \
        --d_model 256 \
        --lr 3e-4

Checkpoints are saved to *checkpoint_dir* after every epoch.
The best checkpoint (lowest validation loss) is saved separately as
``best_model.pt``.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow ``python src/train.py`` to import from the src/ package
sys.path.insert(0, os.path.dirname(__file__))

from dataset import get_dataloader
from model import Seq2SeqTransformer
from tokenizer import load_tokenizer, PAD_ID, BOS_ID, EOS_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimiser, device, clip_grad: float = 1.0):
    """Run a single training epoch.

    Returns:
        Average cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing: decoder input is tgt[:-1], labels are tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        logits = model(src, tgt_input)  # (batch, tgt_len-1, vocab)

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = tgt_labels.reshape(-1)

        loss = criterion(logits_flat, labels_flat)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()

        # Accumulate loss weighted by the number of non-padding tokens
        num_tokens = (labels_flat != PAD_ID).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model on *loader*.

    Returns:
        Average cross-entropy loss.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        logits = model(src, tgt_input)

        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = tgt_labels.reshape(-1)

        loss = criterion(logits_flat, labels_flat)

        num_tokens = (labels_flat != PAD_ID).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def save_checkpoint(path, model, optimiser, epoch, val_loss, config):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "config": config,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Data ---
    print("[train] Loading datasets …")
    train_loader = get_dataloader(
        args.train_src,
        args.train_tgt,
        args.sp_model,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        args.val_src,
        args.val_tgt,
        args.sp_model,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sp = load_tokenizer(args.sp_model)
    vocab_size = sp.get_piece_size()
    print(f"[train] Vocabulary size: {vocab_size}")
    print(f"[train] Training batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # --- Model ---
    config = dict(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=PAD_ID,
    )
    model = Seq2SeqTransformer(**config).to(device)
    print(f"[train] Trainable parameters: {count_parameters(model):,}")

    # --- Optimiser, loss, scheduler ---
    optimiser = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimiser, device, args.clip_grad)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f} (ppl={train_ppl:.1f})  "
            f"val_loss={val_loss:.4f} (ppl={val_ppl:.1f})  "
            f"lr={optimiser.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save per-epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint(ckpt_path, model, optimiser, epoch, val_loss, config)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            save_checkpoint(best_path, model, optimiser, epoch, val_loss, config)
            print(f"  ✓ New best model saved  (val_loss={best_val_loss:.4f})")

    print("[train] Training complete.")
    print(f"[train] Best validation loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train the JA→EN Transformer")

    # Data
    p.add_argument("--train_src", required=True, help="Training source file (Japanese)")
    p.add_argument("--train_tgt", required=True, help="Training target file (English)")
    p.add_argument("--val_src", required=True, help="Validation source file (Japanese)")
    p.add_argument("--val_tgt", required=True, help="Validation target file (English)")
    p.add_argument("--sp_model", required=True, help="SentencePiece model file")
    p.add_argument("--max_len", type=int, default=128, help="Max tokens per sentence")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="checkpoints")

    # Model architecture
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=3)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
