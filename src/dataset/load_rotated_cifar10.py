import torch
from torch.utils.data import TensorDataset, Dataset
from utils import pickle_load
import os


def load_rotated_cifar10(data_dir):
    path = os.path.join(data_dir, 'rotated-cifar10.pkl')
    train_tensors = pickle_load(path)

    train_datasets = {}


    for cid in train_tensors.keys():
        train_datasets[cid] = {}

        tensors = train_tensors[cid]

        for part in ['train', 'test']:
            train_datasets[cid][part] = {}

            tensor = tensors[part]

            X = tensor['X']
            Y = tensor['Y']

            dataset = TensorDataset(X, Y)
            train_datasets[cid][part] = dataset

    return train_datasets