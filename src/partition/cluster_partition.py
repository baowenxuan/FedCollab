"""
Artificial Cluster Partition

Each cluster get 10 clusters,
"""

import numpy as np

from .utils import get_labels


def cluster4_partition(dataset, num_labels, num_clients, k=0.5):
    """
    Partition the data into four clusters
    """
    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    assert num_clients % 4 == 0
    assert num_labels == 10

    num_clients_per_cluster = num_clients // 4

    cluster_matrix = np.array([
        [k, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.5, k, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, k, 0.5, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, k]
    ])

    # expand the cluster matrix to client matrix
    client_matrix = np.repeat(cluster_matrix, num_clients_per_cluster, axis=0)
    client_matrix = client_matrix / num_clients_per_cluster

    # cumulative matrix
    cumulate = client_matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.49).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for cid in range(num_clients):
        idxs = []
        for label in range(num_labels):
            idxs.append(idxs_by_label[label][cumulate[cid, label]:cumulate[cid + 1, label]])

        partition_idxs[cid] = np.concatenate(idxs)

    return partition_idxs
