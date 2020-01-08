from torch.utils import data
import torch
import numpy as np


class ToyDataset(data.Dataset):

    def __init__(self, n_examples):
        self.n_examples = n_examples

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        r = np.random.choice([0, 1])
        v = np.random.randn() + r
        u = np.random.randn() + v
        w = np.random.randn() + v
        x = torch.tensor([r, u])
        y = torch.tensor([float(w > 0)])
        z = torch.tensor([float(r)])
        return x, y, z
