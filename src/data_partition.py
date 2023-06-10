"""
Partition a centralized dataset to clients in FL
"""

import os
import argparse

import torch
from torch.utils.data import ConcatDataset
import numpy as np

from dataset import create_dataset, shapes_out
from partition import create_partition
from utils import pickle_save


# from visual import visualize_label_distribution


def main(args):
    """
    :return: client_sample_id
    {cid: {'train': list of sample ids}}
    """

    # get dataset
    datasets = create_dataset(args.dataset, args.data_dir)

    # pre-process dataset
    dataset = ConcatDataset(datasets)

    # get partition of datasets
    train_client_sample_id, test_client_sample_id = create_partition(dataset, args)

    client_sample_id = (train_client_sample_id, test_client_sample_id)

    # save partition results

    filename = '_'.join(['client', str(args.num_clients),
                         'partition', args.partition,
                         'subsample', args.subsample,
                         'seed', str(args.seed)]) + '.pkl'

    filename = os.path.join(args.partition_dir, args.dataset, filename)

    pickle_save(obj=client_sample_id, file=filename, mode='wb')

    print('Data partition is saved to:', filename)


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset name')

    parser.add_argument('--num_clients', type=int, default=20,
                        help='number of clients')

    # partition

    parser.add_argument('--partition', type=str, default='dir_1',
                        help='how to partition dataset to clients, in format ${method}_${parameters}')

    # subsample

    parser.add_argument('--subsample', type=str, default='none',
                        help='how to subsample, in format ${method}_${parameters}')

    # train-test split

    parser.add_argument('--client_holdout', type=float, default=0,
                        help='fraction of testing clients')

    parser.add_argument('--data_holdout', type=float, default=0,
                        help='fraction of testing samples in training clients')

    # to control randomness
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use')

    # directories
    parser.add_argument('--data_dir', type=str, default='~/data',
                        help='dirname of the dataset')

    parser.add_argument('--partition_dir', type=str, default='~/data/partition',
                        help='dirname to save partition results')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='whether to visualize the data distribution')

    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    args.partition_dir = os.path.expanduser(args.partition_dir)

    args.partition_name, *args.partition_parameters = args.partition.split('_')

    # set default data holdout rate if it is not defined.
    if args.dataset in ['cifar10', 'cifar100', 'coarse-cifar100'] and args.data_holdout == 0:
        args.data_holdout = 1 / 6
    elif args.dataset in ['mnist', 'fmnist'] and args.data_holdout == 0:
        args.data_holdout = 1 / 7

    args.num_labels = shapes_out[args.dataset]

    return args


def set_seed(seed):
    np.random.seed(seed)
    # torch.random.seed(seed)


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.seed)
    main(args)
