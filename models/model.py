import argparse
import os

import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Predictor, self).__init__()
        if num_classes == 2:
            self.linear_layer = nn.Linear(num_features, 1)
            self.output_layer = nn.Sigmoid()
        else:
            self.linear_layer = nn.Linear(num_features, num_classes)
            self.output_layer = nn.Softmax()

    def forward(self, x):
        logits = self.linear_layer(x)
        predictions = self.output_layer(logits)
        return logits, predictions


class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.c = nn.Parameter(torch.ones(1))
        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(3, 1)))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits, targets):
        s = torch.sigmoid((1 + torch.abs(self.c)) * logits)
        z_hat = torch.tensor([s,s*targets,s*(1-targets)]).reshape(-1,3) @ self.w2 + self.b
        return z_hat
