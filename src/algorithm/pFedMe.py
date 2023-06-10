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


class pFedMeServer(LocalFinetuneServer):

    def __init__(self, client_datasets, args):
        BaseServer.__init__(self, client_datasets, args)

        # set hyperparameters
        assert args.gm_opt == 'sgd'
        self.gm_lr = args.gm_lr
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_clients * args.part_rate))

        # init clients
        self.clients = {cid: pFedMeClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # model
        self.model = create_model(args)

        # pFedMe is personalized federated learning
        self.is_pfl = True


class pFedMeClient(LocalFinetuneClient):
    """
    Client of pFedMe
    """

    def __init__(self, cid, datasets, args):
        FedAvgClient.__init__(self, cid, datasets, args)
        self.PM_state = None  # each client saves a personalized model state


    def local_train(self, model, args, dataset='train'):
        """
        Local train
        """

        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        # -------- -------- Outer Loop Hyperparameters -------- --------
        # the original paper fix R, the number of local updates
        # to match with other algorithms, here we fix the number of epochs
        # notice that R = num_epochs * num_data / batch_size
        num_epochs = args.lm_epochs
        local_lr = args.lm_lr

        # -------- -------- Inner Loop Hyperparameters -------- --------
        per_optimizer = create_optimizer(model, args.lm_opt, args.pfedme_pm_lr)
        K = args.pfedme_K  # number of inner loop to find personalzed solution
        lamda = args.pfedme_lambda  # local regularization term

        # ======== ======== Prepare for Training ======== ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        local_state = deepcopy(model.updated_state_dict())
        local_tensor = state_to_tensor(local_state)

        # -------- -------- OUTER LOOP -------- --------
        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # start from the current local model (LM), and
                model.load_state_dict(local_state, strict=False)
                ref_tensor = model.trainable_parameter_tensor().detach().clone()  # w_{i, r}^t

                # -------- -------- INNER LOOP START -------- --------

                # train for K steps and get the personalized model (PM)
                for i in range(K):
                    # get prediction
                    logits = model(*X)
                    classifier_loss = loss_func(logits, Y)
                    reg_loss = torch.square(torch.norm(model.trainable_parameter_tensor() - ref_tensor))
                    loss = classifier_loss + (lamda / 2) * reg_loss

                    loss.backward()
                    per_optimizer.step()
                    per_optimizer.zero_grad()

                    with torch.no_grad():
                        # record the loss and accuracy
                        num_examples = len(X[0])
                        total_examples += num_examples

                        total_loss += loss.item() * num_examples

                        metric = metric_func(logits, Y)
                        total_metric += metric.item() * num_examples

                # -------- -------- INNER LOOP END -------- --------

                per_state = model.updated_state_dict()
                self.PM_state = deepcopy(per_state)
                per_tensor = state_to_tensor(per_state)

                local_tensor = local_tensor - local_lr * lamda * (local_tensor - per_tensor)
                local_state = tensor_to_state(local_tensor, local_state)

        model.load_state_dict(local_state, strict=False)  # the local state is uploaded

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data


    def local_personalize(self, model, args, dataset='train'):
        model.load_state_dict(self.PM_state, strict=False)

        return 0, 0, 1
