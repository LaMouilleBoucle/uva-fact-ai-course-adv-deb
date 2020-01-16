import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class UTKFace(Dataset):
    def __init__(self, directory, protected_vars=[]):
        self.images = []
        self.vars = []
        self.labels = []
        for idx, imgdir in enumerate(os.listdir(directory)):
            with open(directory + '/' + imgdir, 'rb') as img:
                self.images.append(Image.open(img).convert('RGB'))
            self.labels.append(imgdir.split('_')[0])
            self.vars.append(imgdir.split('_')[1:3])
        print(self.images[0])
        print(self.vars[0])
        print(self.labels[0])

    def one_hot_encode(self, data):
        encoder = LabelEncoder().fit(data)
        one_hot_idx = encoder.transform(data)
        one_hot = np.eye(len(encoder.classes_))[one_hot_idx]

        return encoder, one_hot

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.protected[idx]

if __name__ == '__main__':

    batch_size = 10
    data = UTKFace('./UTKFace')

    train_data, test_data = (Subset(data, range(0, data.lengths[0])), Subset(data, range(data.lengths[0], data.lengths[0]+data.lengths[1]+1)) )
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    for i, (batch, protected, labels) in enumerate(loader):
        print(batch[0])
        print(batch.shape)
        print(protected.shape)
        print(labels.shape)
        if i > -1:
            break
    for i, (batch, protected, labels) in enumerate(train_loader):
        print(batch[0])
        print(batch.shape)
        print(protected.shape)
        print(labels.shape)
        if i > -1:
            break
    for i, (batch, protected, labels) in enumerate(test_loader):
        print(batch.shape)
        print(protected.shape)
        print(labels.shape)
        if i > -1:
            break
