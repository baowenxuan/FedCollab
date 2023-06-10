import torch
import numpy as np
import random

from dataset import create_fed_dataset
from collab import load_collab
from algorithm import create_system
from utils import pickle_load, pickle_save
from options import args_parser


def main(args):
    # initialize federated dataset
    train_datasets = create_fed_dataset(args)

    collab = load_collab(args, train_datasets)

    histories = []

    for idx, coalition in enumerate(collab):
        print('Running Coalition %d / %d' % (idx + 1, len(collab)))
        print(coalition)

        sub_datasets = {cid: dataset for (cid, dataset) in train_datasets.items() if cid in coalition}

        setup_seed(args.seed)
        server = create_system(args.algorithm, sub_datasets, args)
        # print(server.model.state_dict())
        server.run(args)

        histories.append(server.history.data)

    content = {
        'args': args,
        'collab': collab,
        'histories': histories,
    }
    if args.history_path != 'none':
        pickle_save(content, args.history_path, mode='ab')


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
