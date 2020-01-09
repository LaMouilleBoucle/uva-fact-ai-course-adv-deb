import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class AdultUCI(Dataset):
    def __init__(self ,directory, protected_vars):
        
        self.var_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        self.protected_var_names = protected_vars
        self.real_var_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.data = []
        self.labels = []
        self.protected = []
        self.encoder = {}

        with open(directory, 'r') as data:
            self.data = pd.read_csv(directory, names=self.var_names)  

        for idx, name in enumerate(self.var_names):
            if idx == 0:
                data_temp = np.expand_dims(self.data[name].values, axis=1)
                continue
            if name not in self.real_var_names:
                temp_enc = LabelEncoder()
                self.encoder[name] = temp_enc.fit(self.data[name])
                one_hot_idx = self.encoder[name].transform(self.data[name])
                one_hot = np.eye(len(self.encoder[name].classes_))[one_hot_idx]
                if name is 'income':
                    self.labels = torch.tensor(one_hot)
                    continue
                elif name in self.protected_var_names:
                    if 'protected_temp' in locals(): 
                        np.append(protected_temp, one_hot, axis=1)
                    else:
                        protected_temp = one_hot
                    continue
                data_temp = np.append(data_temp, one_hot, axis=1)
            else:
                data_temp = np.append(data_temp, np.expand_dims(self.data[name].values, axis=1), axis=1)
        
        self.data = torch.tensor(data_temp)
        self.protected = torch.tensor(protected_temp)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.protected[idx], self.labels[idx]

if __name__ == '__main__':

    batch_size = 10
    train_data = AdultUCI('./data/adult.data', ['race', 'sex'])
    test_data = AdultUCI('./data/adult.test', ['race', 'sex'])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    for i, (batch, protected, labels) in enumerate(train_loader):
        print(batch.shape)
        print(protected.shape)
        print(labels.shape)
        if i > 2:
            break

#test: 16282
#train: 32561
#vars: 14
#out: 1

