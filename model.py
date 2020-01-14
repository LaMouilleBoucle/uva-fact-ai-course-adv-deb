import argparse
import os

import torch
import torch.nn as nn
import numpy as np

class Predictor(nn.Module):
    def __init__(self, num_features):
        super(Predictor, self).__init__()

        self.linear_layer = nn.Linear(num_features, 1)
        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear_layer(x)
        predictions = self.output_layer(logits)

        return logits, predictions


class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()

        self.c = nn.Parameter(torch.ones(1))

        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(3,1)))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits, targets):
        s = torch.sigmoid((1 + torch.abs(self.c)) * logits)
        z_hat = (torch.cat((s, s*targets, s*(1-targets)),dim=1) @ self.w2 + self.b)

        return z_hat, torch.sigmoid(z_hat)
