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


class PerFedAvgServer(LocalFinetuneServer):
    """
    Server of PerFedAvg (FO)
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
        self.clients = {cid: PerFedAvgClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # model
        self.model = create_model(args)

        # Local finetuning is personalized federated learning
        self.is_pfl = True


class PerFedAvgClient(LocalFinetuneClient):
    """
    Client of PerFedAvg (FO)
    """
    def __init__(self, cid, datasets, args):
        FedAvgClient.__init__(self, cid, datasets, args)
        self.finetune_steps = 1

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

        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                if len(Y) < 4:
                    continue  # skip too small batch

                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                pivot = len(Y) // 2
                X1 = [x[:pivot] for x in X]
                X2 = [x[pivot:] for x in X]
                Y1, Y2 = Y[:pivot], Y[pivot:]

                state = deepcopy(model.updated_state_dict())

                # get prediction
                logits = model(*X1)
                loss = loss_func(logits, Y1)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # get prediction
                logits = model(*X2)
                loss = loss_func(logits, Y2)

                loss.backward()

                model.load_state_dict(state, strict=False)
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # record the loss and accuracy
                    num_examples = len(X2[0])
                    total_examples += num_examples

                    total_loss += loss.item() * num_examples

                    # print(logits.shape, Y.shape)

                    metric = metric_func(logits, Y2)
                    total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data








