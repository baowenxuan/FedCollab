"""
Reference:
    Canh T. Dinh, Nguyen Hoang Tran, Tuan Dung Nguyen:
    Personalized Federated Learning with Moreau Envelopes. NeurIPS 2020
Implementation:
    https://github.com/CharlieDinh/pFedMe/
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
from .LocalFinetune import LocalFinetuneServer, LocalFinetuneClient


class DittoServer(LocalFinetuneServer):

    def __init__(self, client_datasets, args):
        BaseServer.__init__(self, client_datasets, args)

        # check or set hyperparameters
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_clients * args.part_rate))

        # init clients
        self.clients = {cid: DittoClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # model
        self.model = create_model(args)

        # FedAvg is not personalized federated learning
        self.is_pfl = False

class DittoClient(FedAvgClient):
    def __init__(self, cid, datasets, args):
        super(DittoClient, self).__init__(cid, datasets, args)
        self.mu = args.ditto_lambda
        self.PM_state = None  # each client saves a personalized model state

    def local_train(self, model, args, dataset='train'):

        if self.PM_state is None:  # first round
            self.PM_state = deepcopy(model.updated_state_dict())

        # backup newest global model
        global_tensor = model.trainable_parameter_tensor().detach().clone()

        # train global model
        avg_loss, avg_metric, num_data = FedAvgClient.local_train(self, model, args, dataset='train')

        # backup trained global model
        LM_state = deepcopy(model.updated_state_dict())

        # train personalized model
        model.load_state_dict(self.PM_state)
        avg_loss2, avg_metric2, num_data2 = self.train_personalized_model(model, global_tensor, args, 'train')

        # save PM
        self.PM_state = deepcopy(model.updated_state_dict())

        # load back trained global model
        model.load_state_dict(LM_state, strict=False)

        return avg_loss, avg_metric, num_data


    def train_personalized_model(self, model, global_tensor, args, dataset='train'):
        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs

        # ======== ======== Prepare for Training ======== ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

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
        model.load_state_dict(self.PM_state, strict=False)

        return 0, 0, 1
