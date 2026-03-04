import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.sft.dataset import build_sft_dataset, sft_collate_fn
from src.model.transformer import Transformer
from src.train.optim import build_optimizer_muon, build_scheduler
from src.train.sft_loop import sft_train_loop
from src.utils.config import load_lm_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-interval-tokens", type=int, default=50_000_000)
    parser.add_argument("--checkpoint-prefix", type=str, default="sft")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-name", type=str, default="sft_final.pt")
    args = parser.parse_args()

    config = load_lm_config()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT training")

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"], use_fast=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    dataset = build_sft_dataset(config=config)

    batch_size = int(args.batch_size) if args.batch_size is not None else int(config["batch_size"])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=int(pad_token_id)),
    )

    model = Transformer(
        vocab_size=int(config["vocab_size"]),
        dim_model=int(config["dim_model"]),
        dim_k=int(config["dim_k"]),
        num_q_heads=int(config["num_q_heads"]),
        group_size=int(config["group_size"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        intermediate_size=int(config["intermediate_size"]),
        eps=float(config["eps"]),
        dropout=float(config["dropout"]),
        long_context=bool(config.get("long_context", False)),
    )

    ckpt = torch.load(str(args.init_from), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained weights from: {args.init_from}")

    if "cuda" in str(device).lower():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.compile:
        model = torch.compile(model, options={"triton.cudagraphs": False})

    optimizer = build_optimizer_muon(
        model,
        muon_lr=float(args.muon_lr),
        adamw_lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    total_steps = max(1, len(loader))

    scheduler = build_scheduler(optimizer, total_steps, warmup_ratio=float(args.warmup_ratio))

    checkpoint_dir = str(config["checkpoint_output_dir"])
    model_output_dir = str(config["model_output_dir"])

    os.makedirs(model_output_dir, exist_ok=True)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(
        "SFT config | "
        f"device={device} "
        f"batch_size={batch_size} "
        f"micro_batch_size={args.micro_batch_size} "
        f"total_steps={total_steps} "
        f"lr={float(args.lr):.3e} "
        f"muon_lr={float(args.muon_lr):.3e}"
    )

    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "wandb is not installed. Install it with 'pip install wandb' to run SFT training."
        ) from e

    init_from_name = os.path.splitext(os.path.basename(str(args.init_from)))[0]
    wandb_run = wandb.init(
        project="sft",
        name=f"sft_{init_from_name}",
        config={
            "init_from": str(args.init_from),
            "batch_size": int(batch_size),
            "micro_batch_size": (int(args.micro_batch_size) if args.micro_batch_size is not None else None),
            "lr": float(args.lr),
            "muon_lr": float(args.muon_lr),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "max_grad_norm": float(args.max_grad_norm),
            "total_steps": int(total_steps),
            "checkpoint_interval_tokens": int(args.checkpoint_interval_tokens),
            "checkpoint_dir": str(checkpoint_dir),
            "model_output_dir": str(model_output_dir),
        },
    )

    sft_train_loop(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        micro_batch_size=args.micro_batch_size,
        max_grad_norm=float(args.max_grad_norm),
        use_amp=True,
        log_every=int(args.log_every),
        pad_token_id=int(pad_token_id),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_tokens=int(args.checkpoint_interval_tokens),
        checkpoint_prefix=str(args.checkpoint_prefix),
        wandb_run=wandb_run,
    )

    save_path = os.path.join(model_output_dir, str(args.save_name))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        save_path,
    )
    print(f"Saved final model to: {save_path}")


if __name__ == "__main__":
    main()
