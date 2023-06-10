from torch.utils.data import ConcatDataset, Subset
import numpy as np

from utils import pickle_load
from .torchvision_dataset import create_torchvision_dataset
from .load_rotated_cifar10 import load_rotated_cifar10

from .rotate import RotatedDataset, load_rotation_config
from .swap import LabelSwapDataset, load_labelswap_config


def create_dataset(dataset_name, data_dir):
    """
    Create the dataset and its partition
    :param args:
    :return:
    """
    torchvision_dataset_names = [
        'mnist',
        'fmnist',
        'cifar10',
        'cifar100',
        'coarse-cifar100',
    ]

    if dataset_name in torchvision_dataset_names:
        datasets = create_torchvision_dataset(dataset_name=dataset_name, data_dir=data_dir)

    else:
        raise NotImplementedError('Unknown dataset!')

    return datasets


def create_fed_dataset(args):
    dataset_name = args.dataset
    data_dir = args.data_dir
    partition_path = args.partition_path
    # train_valid_split = args.train_valid_split

    rotation_config = args.rotation_config
    labelswap_config = args.labelswap_config

    if dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'coarse-cifar100']:
        datasets = create_dataset(dataset_name, data_dir)
        dataset = ConcatDataset(datasets)
        train_client_sample_id, _ = pickle_load(partition_path)  # we assume no test clients in cross-silo setting

        if rotation_config != 'none':  # feature shift
            rotate_dict = load_rotation_config(rotation_config)
            train_datasets = {
                cid: {part: RotatedDataset(Subset(dataset, indices), angle=rotate_dict[cid]) for part, indices in
                      sids.items()} for cid, sids in train_client_sample_id.items()
            }
        elif labelswap_config != 'none':  # label swap
            swap_dict = load_labelswap_config(labelswap_config, args.num_labels)
            train_datasets = {
                cid: {part: LabelSwapDataset(Subset(dataset, indices), label_mapper=swap_dict[cid]) for part, indices in
                      sids.items()} for cid, sids in train_client_sample_id.items()
            }

        else:
            train_datasets = {
                cid: {part: Subset(dataset, indices) for part, indices in sids.items()} for cid, sids in
                train_client_sample_id.items()
            }

    elif dataset_name == 'rotated-cifar10':  # load prepared rotated cifar-10
        train_datasets = load_rotated_cifar10(data_dir)

    else:
        raise NotImplementedError

    # if train_valid_split < 1.0:
    #     for cid, datasets in train_datasets.items():
    #         n = len(datasets['train'])
    #         k = int(n * args.train_valid_split)
    #         all_indices = np.arange(n)
    #         np.random.shuffle(all_indices)
    #
    #         datasets['valid'] = Subset(datasets['train'], all_indices[k:])
    #         datasets['train'] = Subset(datasets['train'], all_indices[:k])

    return train_datasets
