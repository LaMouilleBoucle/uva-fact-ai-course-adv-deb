import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torchvision import transforms


class UTKFace(Dataset):
    def __init__(self, directory, protected_vars=[]):
        self.samples = []
        self.vars = []
        self.labels = []
        self.transform = transforms.Compose([transforms.ToTensor()])
        for idx, imgdir in enumerate(os.listdir(directory)):
            self.samples.append(directory + '/' + imgdir)
            self.labels.append(imgdir.split('_')[0])
            self.vars.append(imgdir.split('_')[1:3])

    def one_hot_encode(self, data):
        encoder = LabelEncoder().fit(data)
        one_hot_idx = encoder.transform(data)
        one_hot = np.eye(len(encoder.classes_))[one_hot_idx]

        return encoder, one_hot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as img:
            image = self.transform(Image.open(img).convert('RGB'))

        return image, self.labels[idx], self.vars[idx]

if __name__ == '__main__':

    batch_size = 10
    data = UTKFace('./UTKFace')
    print(len(data))

    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)

    for i, (batch, protected, labels) in enumerate(loader):
        print(batch[0])
        print(batch.shape)
        print(protected.shape)
        print(labels.shape)
        if i > -1:
            break