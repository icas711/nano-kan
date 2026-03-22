"""
Training script for nano-KAN.
Single-GPU training loop with gradient accumulation, mixed precision, and cosine LR.
"""

import os
import sys
import math
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train nano-KAN")
    # Data
    p.add_argument("--data_dir", type=str, default="data/shakespeare")
    # Model
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--kan_grid_size", type=int, default=5)
    p.add_argument("--kan_spline_order", type=int, default=3)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--lr_decay_iters", type=int, default=5000)
    p.add_argument("--weight_decay", type=float, default=2e-1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Logging / eval
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--patience", type=int, default=10, help="Early stopping: stop after N evals without improvement")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="out")
    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1337)

    device = args.device
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map.get(args.dtype, torch.float16)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # ---- Data ----------------------------------------------------------
    train_path = os.path.join(args.data_dir, "train.bin")
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}. Run data/shakespeare/prepare.py first.")
        sys.exit(1)

    # ---- Model ---------------------------------------------------------
    config = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
    )
    model = GPT(config)
    model.to(device)

    if args.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    # ---- Optimizer -----------------------------------------------------
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # Group: weight-decay vs no-decay
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2), fused=(device == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))

    # ---- Training loop -------------------------------------------------
    best_val_loss = float("inf")
    no_improve_count = 0
    t0 = time.time()

    for it in range(args.max_iters):
        # LR schedule
        lr = get_lr(it, args.warmup_iters, args.lr_decay_iters, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, args.data_dir, args.block_size, args.batch_size, device, args.eval_iters)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                no_improve_count = 0
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iter": it,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))
                print(f"  → saved checkpoint (val_loss={best_val_loss:.4f})")
            else:
                no_improve_count += 1
                if args.patience > 0 and no_improve_count >= args.patience:
                    print(f"Early stopping at iter {it} (no improvement for {args.patience} evals)")
                    break

        # Forward / backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.grad_accum_steps):
            X, Y = get_batch("train", args.data_dir, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            lossf = loss.item() * args.grad_accum_steps
            print(f"iter {it}: loss {lossf:.4f}, lr {lr:.6f}, dt {dt*1000:.0f}ms")

    print("Training complete.")


if __name__ == "__main__":
    main()
