import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torchvision import transforms


class UTKFace(Dataset):
    def __init__(self, directory, protected_vars):
        self.samples = []
        self.vars = {'sex': [], 'race': []}
        self.labels = []
        self.protected_var_names = protected_vars
        self.transform = transforms.Compose([transforms.Resize(100), transforms.ToTensor()])
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

        _, self.labels = self.one_hot_encode(pd.cut(self.labels,
                                                    bins=[0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 120],
                                                    # labels=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
                                                    right=True))
        # Count number of people per age category
        # print(self.labels.value_counts())
        # Count number of men vs women 
        # print(f"{self.vars['sex'].count(0)} men.")
        # print(f"{self.vars['sex'].count(1)} women.")
        # Count racial ratios
        # print(f"{self.vars['race'].count(0)} white people.")
        # print(f"{self.vars['race'].count(1)} black people.")
        # print(f"{self.vars['race'].count(2)} asian people.")
        # print(f"{self.vars['race'].count(3)} indian people.")
        # print(f"{self.vars['race'].count(4)} other people.")

        self.labels = torch.tensor(self.labels).float()

        for no, var in enumerate(self.vars):
            if var in self.protected_var_names:
                # 0 is male

                self.protected_vars = torch.tensor([value == 0 for value in self.vars[var]]).float().unsqueeze(dim=1)

        #         _, temp = self.one_hot_encode(self.vars[var])
        #         if no == 0:
        #             varS = temp
        #         else:
        #             varS = np.append(varS, temp, axis=1)
        # self.protected_vars = torch.tensor(varS).float()

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

        return image, self.labels[idx], self.protected_vars[idx]
