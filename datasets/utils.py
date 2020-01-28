import math
import os
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import MaxAbsScaler

from datasets.adult_dataset import AdultUCI
from datasets.communities_crime_dataset import CommunitiesCrimeDataset
from datasets.image_dataset import UTKFace


def load_utkface(base_path, batch_size):
    data_path = os.path.join(base_path, 'data/UTKFace')

    data = UTKFace(data_path, protected_vars=['sex'])
    train_data, test_data = torch.utils.data.random_split(data, [math.ceil(len(data) * 0.6),
                                                                 len(data) - math.ceil(len(data) * 0.6)])
    test_data, val_data = torch.utils.data.random_split(test_data, [math.ceil(len(test_data) * 0.5),
                                                                    len(test_data) - math.ceil(len(test_data) * 0.5)])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def load_communities_crime_dataset(base_path, batch_size):
    cc_dataset = CommunitiesCrimeDataset(os.path.join(base_path,'data/'))
    end_of_train = int(0.7 * len(cc_dataset))
    train_dataset = Subset(cc_dataset, range(0, end_of_train))
    test_dataset = Subset(cc_dataset, range(end_of_train, len(cc_dataset)))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_loader, test_loader


def load_adult_dataset(base_path, batch_size):
    train_path = os.path.join(base_path, 'data/adult.data')
    test_path = os.path.join(base_path, 'data/adult.test')
    adult_dataset = AdultUCI([train_path, test_path], ['sex'])
    end_of_train = int(0.7 * len(adult_dataset))
    min_max_scaler = MaxAbsScaler()
    adult_dataset.data[:end_of_train] = torch.tensor(min_max_scaler.fit_transform(adult_dataset.data[:end_of_train].numpy()))
    adult_dataset.data[end_of_train:] = torch.tensor(min_max_scaler.transform(adult_dataset.data[end_of_train:].numpy()))
    train_dataset = Subset(adult_dataset, range(0, end_of_train))
    test_dataset = Subset(adult_dataset, range(end_of_train, len(adult_dataset)))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_loader, test_loader


def get_dataloaders(batch_size, dataset):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset == 'images':
        train_loader, val_loader, test_loader = load_utkface(base_path, batch_size)
    elif dataset == 'adult':
        train_loader, test_loader = load_adult_dataset(base_path, batch_size)
        val_loader = None
    elif dataset == 'crime':
        train_loader, test_loader = load_communities_crime_dataset(base_path, batch_size)
    return train_loader, val_loader, test_loader
