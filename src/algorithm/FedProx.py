"""
FedProx

Reference:
    Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith:
    Federated Optimization in Heterogeneous Networks. MLSys 2020
Note:
    Official implementation alters the optimizer (maybe for better efficiency). Here to support more optimizer, I add
    the proximal term in the loss instead. They are mathematically equivalent when the optimizer is SGD.
"""

import torch

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .FedAvg import FedAvgServer


class FedProxServer(FedAvgServer):
    """
    Server of FedProx
    Note: run, train and eval are the same as FedAvgServer
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
        self.clients = {cid: FedProxClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # model (placeholder)
        self.model = create_model(args)

        # FedProx is not personalized federated learning
        self.is_pfl = False


class FedProxClient(BaseClient):
    """
    Client of FedProx
    """

    def __init__(self, cid, datasets, args):
        super(FedProxClient, self).__init__(cid, datasets, args)
        self.mu = args.fedprox_mu

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

        global_tensor = model.trainable_parameter_tensor().detach().clone()

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                classifier_loss = loss_func(logits, Y)
                proximal_term = torch.square(torch.norm(model.trainable_parameter_tensor() - global_tensor))
                loss = classifier_loss + (self.mu / 2) * proximal_term

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

    def local_personalize(self, model, args, dataset='train'):
        raise NotImplementedError('FedProx does not have personalization! ')

    def local_eval(self, model, args, dataset='test'):
        """
        Local Evaluation (No Personalization)
        """
        return super(FedProxClient, self).local_eval(model, args, dataset)
