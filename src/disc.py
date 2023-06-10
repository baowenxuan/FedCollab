import os
import argparse
import torch
import numpy as np
import random

from dataset import create_fed_dataset, shapes_in, shapes_out
from collab import DiscEstimateServer
from utils import pickle_save


def main(args):
    # initialize federated dataset
    train_datasets = create_fed_dataset(args)

    # initialize server
    server = DiscEstimateServer(client_datasets=train_datasets, args=args)

    disc_matrix = server.estimate_all(args)

    print('Estimated pairwise divergence')
    print(disc_matrix)

    ms = torch.Tensor([len(dataset['train']) for dataset in train_datasets.values()])
    m = ms.sum()
    beta = ms / m

    # save the divergence to a folder.
    if not args.debug:
        pickle_save((disc_matrix, m, beta, args), args.divergence_path, mode='ab')

        print('Pairwise divergence is saved to:', args.divergence_path)


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset name, [mnist, cifar10]')

    parser.add_argument('--partition_config', type=str, default='client_10_partition_fedavg_2_seed_0',
                        help='pre-defined data partition for each client')

    parser.add_argument('--rotation_config', type=str, default='none',
                        help='pre-defined rotation angle for each client')

    parser.add_argument('--labelswap_config', type=str, default='none',
                        help='pre-defined label swapping for each client')

    parser.add_argument('--divergence', type=str, default='H',
                        help='the type of divergence we want to estimate')

    parser.add_argument('--model', type=str, default='mlp',
                        help='the model type used to estimate distribution distances')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer, [sgd, adam, ...]')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')

    parser.add_argument('--rounds', type=int, default=100,
                        help='number of global communication rounds')

    parser.add_argument('--iters', type=int, default=1,
                        help='number of local iterations')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')

    parser.add_argument('--use_valid', action='store_true', default=False,
                        help='whether to use validation set')

    parser.add_argument('--eval_rounds', type=int, default=10,
                        help='evaluate the divergence every k rounds')

    # to control randomness
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use')

    # training
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train ')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers of dataloader')

    # directories
    parser.add_argument('--data_dir', type=str, default='~/data',
                        help='where the data is stored')

    parser.add_argument('--partition_dir', type=str, default='~/data/partition',
                        help='where the data partition is stored')

    parser.add_argument('--divergence_dir', type=str, default='../divergence',
                        help='where the client config is stored')

    parser.add_argument('--early_stop', type=int, default=-1,
                        help='stop training if the metric does not increase after m evaluations')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode, compute just one divergence, print more information')

    args = parser.parse_args()

    # in/out-dimension of model
    args.shape_in = shapes_in[args.dataset]
    args.shape_out = shapes_out[args.dataset]
    args.num_labels = max(2, args.shape_out)

    args.data_dir = os.path.expanduser(args.data_dir)
    args.partition_dir = os.path.expanduser(args.partition_dir)

    # the path of partition config
    partition_filename = args.partition_config + '.pkl'
    args.partition_path = os.path.join(args.partition_dir, args.dataset, partition_filename)

    # the path of divergence estimation result
    divergence_filename = args.partition_config + '_' + args.divergence + '_seed_' + str(args.seed) + '.pkl'
    args.divergence_path = os.path.join(args.divergence_dir, args.dataset, divergence_filename)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # the below three lines seem not necessary
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    main(args)
