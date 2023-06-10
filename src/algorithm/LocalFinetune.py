"""
Reference:
    #
Implementation:
    #
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


class LocalFinetuneServer(FedAvgServer):
    """
    Server of Local Finetuning
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
        self.clients = {cid: LocalFinetuneClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # model
        self.model = create_model(args)

        # Local finetuning is personalized federated learning
        self.is_pfl = True

    def run(self, args):
        """
        Run the training and testing pipeline
        """
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.train(self.model, args)
            self.eval(self.model, args)

            self.personalize_and_eval(self.model, args)

    # train and eval are the same

    def personalize_and_eval(self, model, args):
        """
        Evaluate the personalized models
        """
        # current global model
        global_state = deepcopy(model.updated_state_dict())

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.clients.items()):
            model.load_state_dict(global_state, strict=False)  # start from global model
            per_loss, per_metric, per_num_data = client.local_personalize(model, args, 'train')
            loss, metric, num_data = client.local_eval(model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        model.load_state_dict(global_state, strict=False)  # do not perturb the global model

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'PM_test_losses': losses,
            'PM_test_metrics': metrics,
            'PM_test_wavg_loss': agg_loss,
            'PM_test_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class LocalFinetuneClient(FedAvgClient):
    """
    Client of FedAvg
    """
    def __init__(self, cid, datasets, args):
        super(LocalFinetuneClient, self).__init__(cid, datasets, args)
        self.finetune_steps = args.finetune_steps

    # local train is the same as FedAvg Client

    def local_personalize(self, model, args, dataset='train'):
        """
        Local Personalization
        """
        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        # num_epochs = args.lm_epochs

        # ======== ======== Prepare for Training ======== ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        iterator = iter(self.dataloaders[dataset])
        for step in range(self.finetune_steps):
            # Get a batch of data (may iterate the dataset for multiple rounds)
            try:
                *X, Y = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloaders[dataset])  # reset iterator, dataset is shuffled
                *X, Y = next(iterator)

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

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data


    def local_eval(self, model, args, dataset='test'):
        """
        Local Evaluation (No Personalization)
        """
        return super(LocalFinetuneClient, self).local_eval(model, args, dataset)
