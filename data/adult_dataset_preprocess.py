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



# class Model(nn.Module):
#     def __init__(self, var_dim, out_dim):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(var_dim, out_dim, bias=True)
#         self.softmax = nn.Softmax()
#     def forward(self, batch):
#         out = self.linear(batch)
#         return self.softmax(out)

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


            
        # to_tensor = []
        # for ind, name in enumerate(self.var_names):
        #     if name in self.real_var_names:
        #         one_hot = one_hot_encode(vals, name)


        # with open(directory, 'r') as data: 
        #     for i, line in enumerate(data):
        #         variables = line.replace(' ', '').replace('\n', '').split(',')
        #         if len(variables) > 1:
        #             self.data.append(variables[0:len(variables)-1])     
        #             self.labels.append(variables[len(variables)-1])
        
        # for ind, name in enumerate(self.var_names):
        #     if name is 'income':
        #         self.labels = self.one_hot_encode(self.labels, name)
        #         continue
        #     try:
        #         check = int(self.data[ind][0])              
        #         data_enc = [torch.tensor(int(row[ind])) for row in self.data]
        #     except:
        #         data_enc = self.one_hot_encode([row[ind] for row in self.data], name)
        # for i in range(len(self.data)):
        #     self.data[i] = torch.flatten(data_enc)
        # print(self.data[0])
        
        #self.data = tensor_data
    # def one_hot_encode(self, vals, var_name):
    #     set_vals = sorted(list(set(vals)))
    #     vals = [[set_vals.index(val)] for val in vals]
    #     one_hot_val = torch.zeros(len(self.data), len(set_vals)).scatter_(1, torch.tensor(vals), 1)
    #     self.var_desc[var_name] = set_vals 
    #     return one_hot_val
