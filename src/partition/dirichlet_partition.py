"""
Dirichlet Partition
Reference:
    Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
Implementation:
    https://github.com/google-research/federated/blob/master/generalization/synthesization/dirichlet.py
Note:
    Here we follow the implementation of the original paper, which might be different with some later papers.
    Notice that
        (1) Each client's label distribution follow a dirichlet distribution. Not each label's client distribution.
        (2) Therefore we need a normalization
        (3) The parameter of label distribution is alpha * prior, not alpha * one_hot.
"""

import numpy as np


from .utils import get_labels


def dirichlet_partition(dataset, num_labels, num_clients, alpha):
    """
    :param dataset: Dataset
    :param num_clients: number of clients ()
    :param alpha: concentration score. Larger alpha -> more IID
    :return:
    """
    EPS = 0.00001

    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    # label skewness: control each client's label distribution (separately)
    prior = num_samples_per_label / num_samples_per_label.sum()

    if alpha == np.inf:
        matrix = np.tile(prior, (num_clients, 1))
    else:
        matrix = np.random.dirichlet(alpha=alpha * prior + EPS, size=num_clients)

    # renormalize to get label distribution matrix
    matrix += EPS
    matrix = matrix / matrix.sum(axis=0)

    # cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.49).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for cid in range(num_clients):
        idxs = []
        for label in range(num_labels):
            idxs.append(idxs_by_label[label][cumulate[cid, label]:cumulate[cid + 1, label]])

        partition_idxs[cid] = np.concatenate(idxs)

    return partition_idxs