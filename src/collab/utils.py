import torch
import numpy as np


def choice_to_collab_list(choice):
    choice = choice.numpy()
    clusters = set(choice)
    collab = []
    for cluster_id in clusters:
        collab.append(list(np.where(choice == cluster_id)[0]))

    return collab


def choice_to_fixed_alpha(choice, beta):
    num_clients = len(choice)
    alpha = torch.zeros(num_clients, num_clients)
    alpha[choice.view(-1, 1) == choice.view(1, -1)] = 1.0
    alpha = alpha * beta
    alpha = alpha / alpha.sum(dim=1, keepdims=True)
    return alpha
