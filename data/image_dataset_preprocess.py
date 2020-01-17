import os
import math
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torchvision import transforms


class UTKFace(Dataset):
    def __init__(self, directory, protected_vars):
        self.samples = []
        self.vars = {'sex':[], 'race':[]}
        self.labels = []
        self.protected_vars = protected_vars
        self.transform = transforms.Compose([transforms.ToTensor()])
        skipped = 0
        for idx, imgdir in enumerate(os.listdir(directory)):
            if len(imgdir.split('_')) < 4:
                skipped += 1
                continue
            self.samples.append(directory + '/' + imgdir)
            self.labels.append(int(imgdir.split('_')[0]))
            self.vars['sex'].append(int(imgdir.split('_')[1]))
            self.vars['race'].append(int(imgdir.split('_')[2]))
        print(f'Skipped {skipped} images.')
        
        for no, var in enumerate(self.vars):
            if var in self.protected_vars:
                _, temp = self.one_hot_encode(self.vars[var])
                if no == 0:
                    varS = temp
                else:
                    varS = np.append(varS, temp, axis=1)
        self.vars = varS
        

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

        return image, self.vars[idx], self.labels[idx]

if __name__ == '__main__':

    batch_size = 10
    data = UTKFace('./UTKFace', protected_vars=['sex'])
    train_data, test_data = torch.utils.data.random_split(data, [math.ceil(len(data)*0.75), len(data) - math.ceil(len(data)*0.75)])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f'Train set is {len(train_data)} images; test set is {len(test_data)}.')

    for i, (batch, protected, labels) in enumerate(train_loader):
        print(batch[0])
        print(batch.shape)
        print(protected)
        print(labels)
        if i > -1:
            break