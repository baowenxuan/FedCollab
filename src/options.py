import os
import argparse
import torch

from dataset import shapes_in, shapes_out


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'coarse-cifar100', 'femnist', 'rotated-cifar10'],
                        help='dataset name')

    parser.add_argument('--partition_config', type=str, default='client_10_partition_fedavg_2_seed_0',
                        help='pre-defined data partition for each client')

    parser.add_argument('--rotation_config', type=str, default='none',
                        help='pre-defined rotation angle for each client')

    parser.add_argument('--labelswap_config', type=str, default='none',
                        help='pre-defined label swapping for each client')

    parser.add_argument('--train_valid_split', type=float, default=1.0,
                        help='use only a subset of data for training')

    parser.add_argument('--collab_config', type=str, default='global',
                        help='the solved collaboration')

    parser.add_argument('--model', type=str, default='linear',
                        help='federated learning model')

    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'bce'],
                        help='loss function')

    parser.add_argument('--metric', type=str, default='acc',
                        choices=['acc', 'bacc'],
                        help='metric function')

    parser.add_argument('--algorithm', type=str, default='fedavg',
                        help='the federated learning algorithm')

    # Global model

    parser.add_argument('--gm_opt', type=str, default='sgd',
                        help='global model optimizer')

    parser.add_argument('--gm_lr', type=float, default=1.0,
                        help='learning rate of global model optimizer')

    parser.add_argument('--gm_rounds', type=int, default=100,
                        help='number of global communication rounds')

    parser.add_argument('--part_rate', type=float, default=1.0,
                        help='client participation rate in each communication rounds')

    # Local model

    parser.add_argument('--lm_opt', type=str, default='sgd',
                        help='local model optimizer')

    parser.add_argument('--lm_lr', type=float, default=0.1,
                        help='learning rate of the local model optimizer')

    parser.add_argument('--lm_epochs', type=int, default=1,
                        help='number of local training epochs, each epoch iterates the local dataset once')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')

    # More Hyperparameters for other algorithms

    # ======== ======== Global Federated Learning ======== ========

    # FedProx
    parser.add_argument('--fedprox_mu', type=float, default=0.1,
                        help='local regularization for FedProx')

    # ======== ======== Finetuning-based ======== ========

    # For local finetuning
    parser.add_argument('--finetune_steps', type=int, default=2,
                        help='number of local finetuning steps for Local Finetune')

    # For pFedMe
    parser.add_argument('--pfedme_lambda', type=float, default=1,
                        help='local regularization for pFedMe')

    parser.add_argument('--pfedme_pm_lr', type=float, default=0.1,
                        help='learning rate of the personalized model for pFedMe')

    parser.add_argument('--pfedme_K', type=int, default=5,
                        help='number of iters for training personalized model for pFedMe')

    # For Ditto
    parser.add_argument('--ditto_lambda', type=float, default=1,
                        help='local regularization for Ditto')

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

    parser.add_argument('--collab_dir', type=str, default='../collab',
                        help='where the client config is stored')

    parser.add_argument('--history_path', type=str, default='../history/default.pkl')

    args = parser.parse_args()

    # in and out-dimension of model
    args.shape_in = shapes_in[args.dataset]
    args.shape_out = shapes_out[args.dataset]
    args.num_labels = max(2, args.shape_out)

    args.data_dir = os.path.expanduser(args.data_dir)
    args.partition_dir = os.path.expanduser(args.partition_dir)
    args.collab_dir = os.path.expanduser(args.collab_dir)

    # the path of partition config
    partition_filename = args.partition_config + '.pkl'
    args.partition_path = os.path.join(args.partition_dir, args.dataset, partition_filename)

    # the path of divergence estimation result
    collab_filename = args.partition_config + '_' + args.collab_config + '.pkl'
    args.collab_path = os.path.join(args.collab_dir, args.dataset, collab_filename)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    return args
