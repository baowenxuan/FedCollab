import numpy as np
from torch.utils.data import Subset

from .utils import get_labels


def cluster_exponential_subsample_rate(num_clients, num_clusters, decay=1.0):
    x = np.exp(- np.arange(num_clusters) * decay)
    q = np.repeat(x, num_clients // num_clusters)
    return q


def subsample_rate(num_clients, subsample_config):
    # parse subsample method and parameters
    alg, *params = subsample_config.split('_')

    if alg == 'none':
        q = np.ones(num_clients)

    elif alg == 'clusterexp':
        num_clusters = int(params[0])
        decay = float(params[1])
        q = cluster_exponential_subsample_rate(num_clients, num_clusters, decay)

    else:
        raise NotImplementedError

    return q


def stratified_subsample(dataset, idxs, num_labels, rate):
    """
    Subsample {rate} ratio of data samples from the subdataset of {dataset[idxs]}
    while keeping the label distribution unchanged
    (random sampling can perturb the label distribution)
    """
    if idxs is None:
        subdataset = dataset
        idxs = np.arange(len(dataset))
    else:
        subdataset = Subset(dataset, idxs)

    labels, idxs_by_label, num_samples_per_label = get_labels(subdataset, num_labels)

    subidxs = []
    for label in range(num_labels):
        subidxs.append(idxs_by_label[label][:round(num_samples_per_label[label] * rate)])

    subidxs = np.concatenate(subidxs)

    return idxs[subidxs]

