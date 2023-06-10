import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from utils import pickle_load, pickle_save

from collab import discrete_solver, choice_to_collab_list

"""
For all solvers, 
Input:
    m: torch.Tensor, shape = (N, )
    d: torch.Tensor, shape = (N, N)
Output: 
    
"""


def main(args):
    disc, m, beta, *_ = pickle_load(args.divergence_path, multiple=True)[0]

    print(_)
    #
    # print(disc)

    disc = torch.max(torch.Tensor([0]), disc)

    if args.visualize:
        plt.imshow(disc)
        plt.show()

    solver = discrete_solver

    # if args.collab_solver == 'discrete':
    #     solver = discrete_solver
    # elif args.collab_solver == 'weighted':
    #     solver = bad_discrete_solver

    low = np.inf
    best = None
    logs = []
    # import time
    # aa = time.time()
    for i in range(100):
        choice, error, log = solver(disc, m, beta, C=args.C)
        if error < low:
            low = error
            best = choice

        logs.append(log)

    # bb = time.time()
    # print(bb - aa)

    choice = best

    collab = choice_to_collab_list(choice)

    print('Solved collaboration:', collab)

    print('Loss:', error * len(beta))

    pickle_save(collab, args.collab_path, 'wb')
    # pickle_save(logs, '../tmp/logs.pkl', 'wb')

    # init = torch.LongTensor([0] * 5 + [1] * 5 + [2] * 10)
    # choice, error = solver(disc, m, beta, C=args.C, init=init, max_iter=0)
    # print(error)
    #
    # choice, error = solver(disc, m, beta, C=args.C, init=init, max_iter=100)
    # print(error)


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset name')

    parser.add_argument('--partition_config', type=str, default='client_10_partition_fedavg_2_seed_0',
                        help='pre-defined data partition for each client')

    parser.add_argument('--divergence_config', type=str, default='H_seed_0',
                        help='the type of divergence we want to estimate, and random seed')

    parser.add_argument('--collab_solver', type=str, default='discrete',
                        help='way to solve collaboration')

    parser.add_argument('--C', type=float, default=1,
                        help='trade off between generalization and dataset shift')

    parser.add_argument('--divergence_dir', type=str, default='../divergence',
                        help='where the client config is stored')

    parser.add_argument('--collab_dir', type=str, default='../collab',
                        help='where the client config is stored')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='whether to visualize divergence')

    # to control randomness
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use')

    args = parser.parse_args()

    divergence_filename = args.partition_config + '_' + args.divergence_config + '.pkl'
    args.divergence_path = os.path.join(args.divergence_dir, args.dataset, divergence_filename)

    collab_filename = '_'.join(
        [args.partition_config, args.divergence_config, args.collab_solver, 'C', str(float(args.C)), 'seed',
         str(args.seed)]) + '.pkl'
    args.collab_path = os.path.join(args.collab_dir, args.dataset, collab_filename)

    return args


if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    main(args)
