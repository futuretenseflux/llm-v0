import yaml

from data.pretraining.training.dataset import BinaryTokenDataset
from data.pretraining.training.sampler import ProportionSampler
from data.pretraining.training.sampling_ratio_generator import DATASET_ORDER, get_sampling_ratios
from model.transformer import Transformer
from torch.utils.data import DataLoader, ConcatDataset
from train.logger import TrainLogger
from train.loop import train_loop
from train.optim import build_optimizer, build_scheduler

with open("configs/lm.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]

SEQ_LENGTH = config["seq_length"]
STRIDE = config.get("stride", None)

dataset_dict = {
    "books" : BinaryTokenDataset(output_dir, "books", SEQ_LENGTH, STRIDE),
    "code" : BinaryTokenDataset(output_dir, "code", SEQ_LENGTH, STRIDE),
    "conv_forum" : BinaryTokenDataset(output_dir, "conv_forum", SEQ_LENGTH, STRIDE),
    "math" : BinaryTokenDataset(output_dir, "math", SEQ_LENGTH, STRIDE),
    "papers" : BinaryTokenDataset(output_dir, "papers", SEQ_LENGTH, STRIDE),
    "primer" : BinaryTokenDataset(output_dir, "primer", SEQ_LENGTH, STRIDE),
    "web" : BinaryTokenDataset(output_dir, "web", SEQ_LENGTH, STRIDE)
}

#ordered by dataset order
datasets = [dataset_dict[d] for d in DATASET_ORDER]
dataset_combined = ConcatDataset(datasets)

TOKEN_BUDGET = 15_000_000_000
NUM_SAMPLES = TOKEN_BUDGET // SEQ_LENGTH
tokens_elapsed = 0

start_ratios = get_sampling_ratios(tokens_elapsed)
sampler = ProportionSampler(datasets, start_ratios, NUM_SAMPLES)

BATCH_SIZE = config["batch_size"]
STEPS = NUM_SAMPLES // BATCH_SIZE
loader = DataLoader(dataset_combined, BATCH_SIZE, sampler=sampler)

model = Transformer(
    vocab_size=config["vocab_size"],
    dim_model=config["dim_model"],
    dim_k=config["dim_k"],
    num_q_heads=config["num_q_heads"],
    group_size=config["group_size"],
    num_decoder_layers=config["num_decoder_layers"],
    intermediate_size=config["intermediate_size"],
    eps=config["eps"],
    dropout=config["dropout"]
)

optimizer = build_optimizer(model, config["learning_rate"], config["optim_weight_decay"])
scheduler = build_scheduler(optimizer, STEPS)

logger = TrainLogger(project="llm-pretrain", run_name="pretrain-run-1", config=config)

train_loop(
    model=model,
    train_loader=loader,
    optimizer=optimizer,
    sampler=sampler,
    device="cuda",
    scheduler=scheduler,
    log_every=100,
    logger=logger,
    use_amp=True,
    max_grad_norm=1.0,
    tokens_elapsed=tokens_elapsed,
    total_steps=STEPS,
    checkpoint_dir=output_dir + "/checkpoints"
)
