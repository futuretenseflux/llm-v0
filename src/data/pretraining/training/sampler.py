from torch.utils.data import Sampler
import torch

class ProportionSampler(Sampler):
    def __init__(self, datasets, probs, num_samples):
        self.datasets = datasets
        self.probs = probs
        self.num_samples = num_samples
        self.lengths = [len(d) for d in datasets]
        self.offsets = torch.cumsum(torch.tensor([0] + self.lengths[:-1]), 0)

    def set_probs(self, probs):
        self.probs = probs

    def __iter__(self):
        for _ in range(self.num_samples):
            d = torch.multinomial(torch.tensor(self.probs), 1).item()
            i = torch.randint(self.lengths[d], (1,)).item()
            yield self.offsets[d] + i
    
    def __len__(self):
        return self.num_samples