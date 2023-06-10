"""
FedNova

Reference：
    Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor:
    Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. NeurIPS 2020
Implementation:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from utils import state_to_tensor, tensor_to_state

from .Base import BaseServer, BaseClient
from .FedAvg import FedAvgServer, FedAvgClient


class FedNovaServer(FedAvgServer):
    """
    Server of FedNova
    """

    def __init__(self, client_datasets, args):
        BaseServer.__init__(self, client_datasets, args)

        # check or set hyperparameters
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_clients * args.part_rate))

        # init clients
        self.clients = {cid: FedNovaClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # model (placeholder)
        self.model = create_model(args)

        # FedProx is not personalized federated learning
        self.is_pfl = False

    def train(self, model, args):
        """
        Train for one communication round
        """
        # current global model
        global_state = deepcopy(model.updated_state_dict())
        global_tensor = state_to_tensor(global_state)

        tensors = []  # local model parameters
        weights = []  # weights (importance) for each client
        losses = []  # training losses for local models (LMs)
        metrics = []  # training metrics (accuracies) for local models (LMs)
        num_steps = []

        # sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.idx2cid[idx] for idx in selected_idxs]

        # iterate randomly selected clients
        for cid in tqdm(selected_cids):
            client = self.clients[cid]
            model.load_state_dict(global_state, strict=False)  # start from global model

            loss, metric, num_data, num_step = client.local_train(model, args, 'train')
            state = deepcopy(model.updated_state_dict())
            tensor = state_to_tensor(state)

            tensors.append(tensor)
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)
            num_steps.append(num_step)

        # train loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'train_selected_idxs': selected_idxs,
            'train_selected_cids': selected_cids,
            'LM_train_losses': losses,
            'LM_train_metrics': metrics,
            'LM_train_wavg_loss': agg_loss,
            'LM_train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

        # model aggregation
        weights = torch.Tensor(weights).to(args.device)
        weights = weights / weights.sum()

        num_steps = torch.Tensor(num_steps).to(args.device)
        avg_steps = (weights * num_steps).sum()

        updates = torch.stack(tensors) - global_tensor
        norm_updates = updates / num_steps.view(-1, 1)
        avg_updates = (weights.view(-1, 1) * norm_updates).sum(dim=0)

        global_tensor = global_tensor + avg_steps * avg_updates
        global_state = tensor_to_state(global_tensor, global_state)  # update global state
        model.load_state_dict(global_state, strict=False)  # start from global model


class FedNovaClient(BaseClient):
    """
    Client of FedNova
    """

    def __init__(self, cid, datasets, args):
        super(FedNovaClient, self).__init__(cid, datasets, args)

    def local_train(self, model, args, dataset='train'):
        """
        Local Training
        """

        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs

        # ======== ======== Prepare for Training ======== ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0
        total_steps = 0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # record the loss and accuracy
                    num_examples = len(X[0])
                    total_examples += num_examples

                    total_loss += loss.item() * num_examples

                    metric = metric_func(logits, Y)
                    total_metric += metric.item() * num_examples

                    total_steps += 1

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data, total_steps

    def local_personalize(self, model, args, dataset='train'):
        """
        Local Personalization
        """
        raise NotImplementedError('GFL does not have personalization! ')
