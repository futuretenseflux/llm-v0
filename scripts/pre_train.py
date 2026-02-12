import yaml

from data.pretraining.training.dataset import BinaryTokenDataset

with open("configs/datasets.yaml", "r") as f:
    config = yaml.safe_load(f)
    output_dir = config["output_dir"]

books = BinaryTokenDataset(output_dir, "books", config["sequence_length"])
code = BinaryTokenDataset(output_dir, "code", config["sequence_length"])
conv_forum = BinaryTokenDataset(output_dir, "conv_forum", config["sequence_length"])
math = BinaryTokenDataset(output_dir, "math", config["sequence_length"])
papers = BinaryTokenDataset(output_dir, "papers", config["sequence_length"])
primer = BinaryTokenDataset(output_dir, "primer", config["sequence_length"])
web = BinaryTokenDataset(output_dir, "web", config["sequence_length"])

datasets = [books, code, conv_forum, math, papers, primer, web]

token_budget = 20_000_000_000

