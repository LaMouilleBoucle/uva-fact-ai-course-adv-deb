import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class AdultUCI(Dataset):
    def __init__(self ,directory):
        
        self.var_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        self.data = []
        self.labels = []
        self.var_desc = {}
        data_enc = []

        
        with open(directory, 'r') as data: 
            for i, line in enumerate(data):
                variables = line.replace(' ', '').replace('\n', '').split(',')
                if len(variables) > 1:
                    self.data.append(variables[0:len(variables)-1])     
                    self.labels.append(variables[len(variables)-1])
        #self.data = np.asarray(self.data)
        print(self.data[0])
        print(self.labels[0])
        for ind, name in enumerate(self.var_names):
            if name is 'income':
                self.labels = self.one_hot_encode(self.labels, name)
                print(type(self.labels))
                continue
            try:
                check = int(self.data[ind][0])              
                data_enc = [torch.tensor(int(row[ind])) for row in self.data]
            except:
                data_enc = self.one_hot_encode([row[ind] for row in self.data], name)
            for i in range(len(self.data)):
                self.data[i][ind] = data_enc[i]
        #self.data = np.asarray(self.data)

        print(self.data[0])
        print(self.labels[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def one_hot_encode(self, vals, var_name):
        set_vals = sorted(list(set(vals)))
        vals = [[set_vals.index(val)] for val in vals]
        one_hot_val = torch.zeros(len(self.data), len(set_vals)).scatter_(1, torch.tensor(vals), 1)
        self.var_desc[var_name] = set_vals 
        return one_hot_val


class Model(nn.Module):
    def __init__(self, var_dim, out_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(var_dim, out_dim, bias=True)
    def forward(self, batch):
        out = self.linear(batch)
        return out

if __name__ == '__main__':

    batch_size = 10
    # for idx, batch in enumerate(AdultUCI('./data/adult.data')):
    #     print(idx)

    train_loader = DataLoader(AdultUCI('./data/adult.data'), batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(AdultUCI('./data/adult.test'), batch_size=batch_size, shuffle=True, num_workers=2)
    model = Model(var_dim=14, out_dim=1)
    for i, (batch, labels) in enumerate(train_loader):
        if i > 1:
            break
        print(batch)
        print(type(batch))
        #batch = torch.tensor(batch)
        print(type(batch))
        print(batch[0])
        print(type(batch[0]))
        out = model(batch)
        print(out)
# country': ['Guatemala', 'Iran', 'Honduras', 'Poland', 'Peru', 'Scotland', 'China', 'Holand-Netherlands', 'United-States', 'Ecuador', 'Germany', 'France', 'Trinadad&Tobago', 'Vietnam', 'India', '?', 'Yugoslavia', 'Nicaragua', 'South', 'Greece', 'Ireland', 'Taiwan', 'Philippines', 'Outlying-US(Guam-USVI-etc)', 'Cambodia', 'Italy', 'Dominican-Republic', 'Columbia', 'Jamaica', 'England', 'Hungary', 'Mexico', 'Portugal', 'Puerto-Rico', 'Hong', 'Japan', 'Laos', 'El-Salvador', 'Thailand', 'Cuba', 'Canada', 'Haiti']
# country': ['Guatemala', 'Iran', 'Poland', 'Honduras', 'Peru', 'Scotland', 'China', 'United-States', 'Ecuador', 'Germany', 'France', 'Trinadad&Tobago', 'Vietnam', 'India', '?', 'Nicaragua', 'Yugoslavia', 'South', 'Greece', 'Ireland', 'Taiwan', 'Philippines', 'Outlying-US(Guam-USVI-etc)', 'Cambodia', 'Italy', 'Dominican-Republic', 'Columbia', 'Jamaica', 'England', 'Hungary', 'Mexico', 'Portugal', 'Puerto-Rico', 'Hong', 'Japan', 'Laos', 'El-Salvador', 'Thailand', 'Cuba', 'Canada', 'Haiti']    

#test: 16282
#train: 32561
#vars: 14
#out: 1

