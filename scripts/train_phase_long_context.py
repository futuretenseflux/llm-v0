import yaml
import torch
import argparse
import os
from src.data.pretraining.first_phase.dataset import BinaryTokenDataset
from src.model.transformer import Transformer
from torch.utils.data import DataLoader
from src.train.logger import TrainLogger
from src.train.loop import train_loop
from src.train.optim import build_optimizer_muon, build_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--stop_loss", type=float)
    parser.add_argument("--token_budget", type=int) # long context budget left
    parser.add_argument("--batch", type=int)
    parser.add_argument("--micro_batch", type=int)
    
    args = parser.parse_args()

    with open("configs/lm.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_dir = config["data_output_dir"] + "/pretraining"

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
        long_context=True # Enable long context training
    )

    state = torch.load(args.path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    def _strip_prefix(sd: dict, prefix: str) -> dict:
        if not isinstance(sd, dict):
            return sd
        if not any(k.startswith(prefix) for k in sd.keys()):
            return sd
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    state = _strip_prefix(state, "_orig_mod.")
    state = _strip_prefix(state, "module.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing keys (first 50):", missing[:50])
        if unexpected:
            print("unexpected keys (first 50):", unexpected[:50])
        raise RuntimeError("Model state_dict did not match current Transformer definition. See printed missing/unexpected keys.")
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model = torch.compile(model, options={"triton.cudagraphs": False})

    stride = config.get("stride", 0.5)
    tokens_elapsed = 0 # for this run

    num_samples = args.token_budget // args.seq_length
    steps = num_samples // (args.batch)

    long_context_dataset = BinaryTokenDataset(dataset_dir, "long_context", args.seq_length, stride)

    loader = DataLoader(
        long_context_dataset,
        batch_size=args.batch,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    optimizer = build_optimizer_muon(
        model, 
        muon_lr=config.get("muon_lr", 0.02),
        adamw_lr=config["learning_rate"],
        weight_decay=config["optim_weight_decay"]
    )
    scheduler = build_scheduler(optimizer, steps)

    logger = TrainLogger(project="llm-long-context-train", run_name="lc-train-run-1", config=config)

    train_loop(
        model=model,
        train_loader=loader,
        device="cuda",
        optimizer=optimizer,
        scheduler=scheduler,
        log_every=10,
        logger=logger,
        use_amp=True,
        micro_batch_size=args.micro_batch,
        max_grad_norm=1.0,
        tokens_elapsed=tokens_elapsed,
        token_budget=args.token_budget,
        stop_loss=args.stop_loss,
        total_steps=steps,
    )

    model_output_dir = config["model_output_dir"]
    os.makedirs(model_output_dir, exist_ok=True)
    save_path = os.path.join(model_output_dir, f"lc_{int(args.seq_length)}.pt")
    model_to_save = getattr(model, "_orig_mod", model)
    torch.save(model_to_save.state_dict(), save_path)

if __name__ == "__main__":
    main()


