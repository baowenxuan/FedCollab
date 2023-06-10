import numpy as np
from torch.utils.data import Subset, ConcatDataset

from .dirichlet_partition import dirichlet_partition
from .cluster_partition import cluster4_partition
from .subsample import subsample_rate
from .utils import stratified_split

from .stat import print_quantity_stat, print_label_distribution_stat


def create_partition(dataset, args, shuffle=False):
    # 1. Standard partition

    num_clients = args.num_clients
    num_labels = args.num_labels
    partition_config = args.partition

    partition_idxs = partition(dataset, num_labels, num_clients, partition_config)

    # ######## ######## ######## ######## ######## ######## ######## ########

    # 3. split (1) training-testing clients and (2) training-testing samples for training clients

    # 3.1. shuffle the clients
    client_ids = list(partition_idxs.keys())
    if shuffle:
        np.random.shuffle(client_ids)

    # subsample rate
    q = subsample_rate(num_clients, args.subsample)

    # 3.2. let (1 - client_holdout) * 100% of them be training clients
    split_pivot = round((1 - args.client_holdout) * args.num_clients)
    train_client_sample_id = {}

    for i in range(split_pivot):
        cid = client_ids[i]
        idxs = partition_idxs[cid]
        np.random.shuffle(idxs)

        split_dict = {
            'train': (1 - args.data_holdout) * q[cid],
            'delete': (1 - args.data_holdout) * (1 - q[cid]),  # this part is not used
            'test': args.data_holdout,
        }

        train_client_sample_id[cid] = stratified_split(dataset, idxs, num_labels, split_dict)

    # 3.3. let the remaining client_holdout * 100% be testing clients
    test_client_sample_id = {}
    for i in range(split_pivot, args.num_clients):
        cid = client_ids[i]
        idxs = partition_idxs[cid]
        np.random.shuffle(idxs)

        # let all samples be testing set
        test_client_sample_id[cid] = {
            'test': idxs,
        }

    # Visualize the final data
    training_idxs = {cid: sids['train'] for cid, sids in train_client_sample_id.items()}
    print('=' * 16, 'TRAINING', '=' * 16)
    print_quantity_stat(training_idxs, args.visualize)
    print_label_distribution_stat(dataset, num_labels, training_idxs, args.visualize)

    testing_idxs = {cid: sids['test'] for cid, sids in train_client_sample_id.items()}
    print('=' * 16, 'TESTING', '=' * 16)
    print_quantity_stat(testing_idxs, args.visualize)
    print_label_distribution_stat(dataset, num_labels, testing_idxs, args.visualize)

    return train_client_sample_id, test_client_sample_id


def partition(dataset, num_labels, num_clients, partition_config):
    """
    Partition a dataset to several clients. However, there is no train-test split or sampling.
    """
    # parse partition method and parameters
    alg, *params = partition_config.split('_')

    # partition

    if alg == 'stratified':
        alpha = np.inf
        partition_idxs = dirichlet_partition(dataset, num_labels, num_clients, alpha)

    elif alg == 'dir':
        alpha = float(params[0])
        partition_idxs = dirichlet_partition(dataset, num_labels, num_clients, alpha)

    elif alg == 'cluster4':
        k = float(params[0])
        partition_idxs = cluster4_partition(dataset, num_labels, num_clients, k)

    else:
        raise NotImplementedError('Unknown data partition algorithm. ')

    return partition_idxs
