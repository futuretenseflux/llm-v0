import os
import yaml
import torch

from src.data.pretraining.first_phase.dataset import BinaryTokenDataset
from src.data.pretraining.first_phase.fim import FIMDataset
from src.data.pretraining.first_phase.sampler import ProportionSampler
from src.data.pretraining.first_phase.sampling_ratio_generator import DATASET_ORDER, get_sampling_ratios
from src.model.transformer import Transformer
from torch.utils.data import DataLoader, ConcatDataset
from src.train.logger import TrainLogger
from src.train.loop import train_loop
from src.train.optim import build_optimizer_muon, build_scheduler
from transformers import AutoTokenizer

with open("configs/lm.yaml", "r") as f:
    config = yaml.safe_load(f)

data_output_dir = config["data_output_dir"]
pretraining_path = data_output_dir + "/pretraining"
checkpoint_output_dir = config["checkpoint_output_dir"]

SEQ_LENGTH = config["seq_length"]
STRIDE = config.get("stride", None)

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"], use_fast=True)
fim_prefix_id = tokenizer.convert_tokens_to_ids("<|fim_prefix|>")
fim_middle_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
fim_suffix_id = tokenizer.convert_tokens_to_ids("<|fim_suffix|>")

if any(tid is None or int(tid) < 0 for tid in [fim_prefix_id, fim_middle_id, fim_suffix_id]):
    raise ValueError("FIM special tokens not found in tokenizer. Did you run scripts/fork_tokenizer.py?")

fim_prob_code = float(config.get("fim_prob_code", 0.5))

dataset_dict = {
    "books" : BinaryTokenDataset(pretraining_path, "books", SEQ_LENGTH, STRIDE),
    "code" : FIMDataset(
        BinaryTokenDataset(pretraining_path, "code", SEQ_LENGTH, STRIDE),
        seq_length=SEQ_LENGTH,
        fim_prefix_id=fim_prefix_id,
        fim_middle_id=fim_middle_id,
        fim_suffix_id=fim_suffix_id,
        fim_prob=fim_prob_code,
        rng_seed=42,
    ),
    "conv_forum" : BinaryTokenDataset(pretraining_path, "conv_forum", SEQ_LENGTH, STRIDE),
    "math" : BinaryTokenDataset(pretraining_path, "math", SEQ_LENGTH, STRIDE),
    "papers" : BinaryTokenDataset(pretraining_path, "papers", SEQ_LENGTH, STRIDE),
    "primer" : BinaryTokenDataset(pretraining_path, "primer", SEQ_LENGTH, STRIDE),
    "web" : BinaryTokenDataset(pretraining_path, "web", SEQ_LENGTH, STRIDE)
}

#ordered by dataset order
datasets = [dataset_dict[d] for d in DATASET_ORDER]
print("All datasets loaded")
dataset_combined = ConcatDataset(datasets)

TOKEN_BUDGET = 16_000_000_000
NUM_SAMPLES = TOKEN_BUDGET // SEQ_LENGTH
tokens_elapsed = 0

start_ratios = get_sampling_ratios(tokens_elapsed)
sampler = ProportionSampler(datasets, start_ratios, NUM_SAMPLES)
print("Sampler created")

BATCH_SIZE = config["batch_size"]
GRAD_ACCUM_STEPS = int(config.get("grad_accum_steps", 1))
MICRO_BATCH_SIZE = config.get("micro_batch_size", None)
STEPS = NUM_SAMPLES // (BATCH_SIZE * GRAD_ACCUM_STEPS)

checkpoint_steps = set()
for denom in [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
    s = STEPS // denom
    if s > 0:
        checkpoint_steps.add(int(s))

loader = DataLoader(
    dataset_combined,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

print("DataLoader created, ", STEPS, " steps", NUM_SAMPLES, " samples")

model = Transformer(
    vocab_size=int(config["vocab_size"]),
    dim_model=int(config["dim_model"]),
    dim_k=int(config["dim_k"]),
    num_q_heads=int(config["num_q_heads"]),
    group_size=int(config["group_size"]),
    num_decoder_layers=int(config["num_decoder_layers"]),
    intermediate_size=int(config["intermediate_size"]),
    eps=float(config["eps"]),
    dropout=float(config["dropout"])
)
model = model.to(device="cuda", dtype=torch.bfloat16)
model = torch.compile(model, options={"triton.cudagraphs": False})
print("Model created")

optimizer = build_optimizer_muon(
    model, 
    muon_lr=config.get("muon_lr", 0.02),
    adamw_lr=config["learning_rate"],
    weight_decay=config["optim_weight_decay"]
)
scheduler = build_scheduler(optimizer, STEPS)
print("Optimizer and scheduler created, starting loop...")

logger = TrainLogger(project="llm-pretrain", run_name="pretrain-run-1", config=config)

train_loop(
    model=model,
    train_loader=loader,
    optimizer=optimizer,
    sampler=sampler,
    device="cuda",
    scheduler=scheduler,
    log_every=10,
    logger=logger,
    use_amp=True,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    micro_batch_size=MICRO_BATCH_SIZE,
    max_grad_norm=1.0,
    tokens_elapsed=tokens_elapsed,
    total_steps=STEPS,
    checkpoint_dir=checkpoint_output_dir,
    checkpoint_steps=checkpoint_steps,
)
