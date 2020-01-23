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
        # return predictions

class ImagePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, drop_probability=0.3):
        super(ImagePredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_probability = drop_probability
        self.convolutions = nn.Sequential(
            nn.Conv2d(self.input_dim, 6, kernel_size=(3,3), stride=1, padding=0),
            # input_dim * 100 * 100 -> 6 * 98 * 98
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
            # 6 * 98 * 98 -> 6 * 48 * 48
            nn.Conv2d(6, 12, kernel_size=(3,3), stride=1, padding=0),
            # 6 * 48 * 48 -> 12 * 46 * 46
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
            # 12 * 46 * 46 -> 12 * 22 * 22
            nn.Dropout(drop_probability),
            nn.Conv2d(12, 24, kernel_size=(3,3), stride=1, padding=0),
            # 12 * 22 * 22 -> 24 * 20 * 20
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
            # 24 * 20 * 20 -> 24 * 9 * 9
            nn.Dropout(drop_probability))
        self.linears = nn.Sequential(
            nn.Linear(24 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
            nn.Sigmoid())
    def forward(self, x):
        convoluted = self.convolutions(x)
        preds = self.linears(convoluted.view(convoluted.size(0), -1))
        return preds

class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()

        self.c = nn.Parameter(torch.ones(1))

        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(3,1)))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits, targets):
        s = torch.sigmoid((1 + torch.abs(self.c)) * logits)
        z_hat = torch.cat((s, s*targets, s*(1-targets)),dim=1) @ self.w2 + self.b

        return z_hat, torch.sigmoid(z_hat)
