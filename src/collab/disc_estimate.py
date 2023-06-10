import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from math import prod
import numpy as np
from itertools import combinations
from tqdm import tqdm
from copy import deepcopy

from model import create_loss, create_metric, create_optimizer
from model import Model
from utils import state_to_tensor, tensor_to_state
from partition import stratified_split

from .model import MLPDivergenceEstimator, LinearDivergenceEstimator, MLPDivergenceEstimatorWithFeatureExtractor


class DiscEstimateServer:
    """
    Server of Domain Discrepancy Estimator
    """

    def __init__(self, client_datasets, args):
        # some useful information
        self.num_clients = len(client_datasets)
        self.idx2cid = {i: cid for i, cid in enumerate(client_datasets)}
        self.cid2idx = {cid: i for i, cid in self.idx2cid.items()}

        # use same number of training data for each client.
        args.min_num_train = min([len(datasets['train']) for datasets in client_datasets.values()])

        self.clients = {cid: DiscEstimateClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        print([len(client.datasets['train']) for client in self.clients.values()])
        print([len(client.datasets['valid']) for client in self.clients.values()])

    def estimate_all(self, args):
        disc_type = args.divergence
        num_rounds = args.rounds
        num_iters = args.iters
        optimizer_name = args.optimizer
        lr = args.lr
        shape_in = args.shape_in
        shape_out = args.shape_out

        num_label = max(2, shape_out)

        if args.model == 'mlp':
            model_class = MLPDivergenceEstimator
        elif args.model == 'linear':
            model_class = LinearDivergenceEstimator
        elif args.model == 'mlpfe_freeze':
            model_class = lambda **kwargs: MLPDivergenceEstimatorWithFeatureExtractor(freeze_extractor=True, **kwargs)
        elif args.model == 'mlpfe_nofreeze':
            model_class = lambda **kwargs: MLPDivergenceEstimatorWithFeatureExtractor(freeze_extractor=False, **kwargs)
        else:
            raise NotImplementedError

        # init models
        if disc_type == 'H':
            model = model_class(feature_dim=shape_in, label_dim=None)  # no label is used
            loss_func = create_loss('bce')
            metric_func = create_metric('bacc')
        elif disc_type == 'C':
            model = model_class(feature_dim=shape_in, label_dim=num_label)
            loss_func = create_loss('bce')
            metric_func = create_metric('bacc')
        elif disc_type == 'W':
            model = model_class(feature_dim=shape_in, label_dim=None, c=0.05)
            loss_func = create_loss('mean')
            metric_func = create_metric('mean')
        else:
            raise NotImplementedError('Unknown discrepancy measure: %s' % disc_type)

        model.to(args.device)

        # initialize an empty pairwise discrepancy matrix
        disc_matrix = torch.zeros((self.num_clients, self.num_clients))

        # initialize a global model
        init_state = deepcopy(model.state_dict())

        # Iterate every pair of clients:
        for i, j in tqdm(combinations(range(self.num_clients), 2)):

            if args.debug:
                i = 12
                j = 15
                print(i, j)

            best_metric = - np.inf
            rounds_no_improve = 0

            # find the two client
            client_i, client_j = self.clients[self.idx2cid[i]], self.clients[self.idx2cid[j]]

            # ######## ######## TRAINING ######## ########

            # start with a same model
            state_g = deepcopy(init_state)

            for r in range(num_rounds):
                # ======== ======== client i ======== ========

                # start from the global model, and a new optimizer
                model.load_state_dict(state_dict=state_g, strict=False)
                optimizer = create_optimizer(model=model, optimizer_name=optimizer_name, lr=lr)

                # train and save state
                loss_i, metric_i = client_i.local_train(model, loss_func, metric_func, optimizer, num_iters, 'train', 0)
                state_i = deepcopy(model.state_dict())

                # ======== ======== client j ======== ========

                # start from the global model, and a new optimizer
                model.load_state_dict(state_dict=state_g, strict=False)
                optimizer = create_optimizer(model=model, optimizer_name=optimizer_name, lr=lr)

                # train and save state
                loss_j, metric_j = client_j.local_train(model, loss_func, metric_func, optimizer, num_iters, 'train', 1)
                state_j = deepcopy(model.state_dict())

                # ======== ======== server ======== ========

                # aggregate and get the new global model
                tensor_i, tensor_j = state_to_tensor(state_i), state_to_tensor(state_j)
                tensor_g = (tensor_i + tensor_j) / 2

                state_g = tensor_to_state(tensor_g, model_state_dict_template=state_g)
                # print(state_g)

                # for Wasserstein distance, we need to modify the model!
                if disc_type == 'W':
                    model.load_state_dict(state_dict=state_g, strict=False)
                    # model.clip()
                    model.normalize()
                    state_g = deepcopy(model.state_dict())

                # evaluation divergence
                if r % args.eval_rounds == 0:
                    model.load_state_dict(state_dict=state_g, strict=False)

                    if args.use_valid:

                        if args.debug:
                            loss_i, metric_i = client_i.local_eval(model, loss_func, metric_func, 'train', 0)
                            loss_j, metric_j = client_j.local_eval(model, loss_func, metric_func, 'train', 1)
                            loss, metric = loss_i + loss_j, metric_i + metric_j
                            tqdm.write("train: %d, %f, %f" % (r, loss_i + loss_j, metric_i + metric_j))

                        loss_i, metric_i = client_i.local_eval(model, loss_func, metric_func, 'valid', 0)
                        loss_j, metric_j = client_j.local_eval(model, loss_func, metric_func, 'valid', 1)
                        loss, metric = loss_i + loss_j, metric_i + metric_j

                        if args.debug:
                            tqdm.write("valid: %d, %f, %f" % (r, loss, metric))

                    else:
                        loss_i, metric_i = client_i.local_eval(model, loss_func, metric_func, 'train', 0)
                        loss_j, metric_j = client_j.local_eval(model, loss_func, metric_func, 'train', 1)
                        loss, metric = loss_i + loss_j, metric_i + metric_j

                        if args.debug:
                            tqdm.write("train: %d, %f, %f" % (r, loss_i + loss_j, metric_i + metric_j))

                    if metric > best_metric:
                        best_metric = metric
                        rounds_no_improve = 0

                    else:
                        rounds_no_improve += 1
                        if 0 < args.early_stop <= rounds_no_improve:
                            break


            # ######## ######## TESTING ######## ########

            # model.load_state_dict(state_dict=state_g, strict=False)
            #
            # loss_i, metric_i = client_i.local_eval(model, loss_func, metric_func, 'valid', 0)
            # loss_j, metric_j = client_j.local_eval(model, loss_func, metric_func, 'valid', 1)
            #
            # loss = loss_i + loss_j
            # metric = metric_i + metric_j

            metric = best_metric

            if args.divergence in ['H', 'C']:
                metric = metric - 1  # save 1/2 divergence

            disc_matrix[i, j] = metric
            disc_matrix[j, i] = metric

            if args.debug:
                break

        return disc_matrix


class DiscEstimateClient:
    """
    Client for Discrepancy Estimator
    """

    def __init__(self, cid, datasets, args):

        # client ID
        self.cid = cid

        #
        self.disc = args.divergence  # ['H', 'C', 'W']
        self.num_classes = max(2, args.shape_out)

        # client local dataset
        self.datasets = {}
        if args.use_valid and 'valid' not in datasets:
            # split the training data to train and valid
            num_total = len(datasets['train'])
            rate_train = (args.min_num_train / 2) / num_total
            part_rate_dict = {
                'train': rate_train,
                'valid': 1 - rate_train
            }

            partition_idxs = stratified_split(dataset=datasets['train'], idxs=None,
                                              num_labels=args.num_labels, part_rate_dict=part_rate_dict)

            train_idxs = partition_idxs['train']
            valid_idxs = partition_idxs['valid']

            self.datasets['train'] = Subset(datasets['train'], indices=train_idxs)
            self.datasets['valid'] = Subset(datasets['train'], indices=valid_idxs)
            self.datasets['test'] = datasets['test']

        else:
            self.datasets = datasets

        # number of training data / testing data / ...
        self.num_data = {key: len(dataset) for key, dataset in self.datasets.items()}
        self.num_data['all'] = sum([len(dataset) for dataset in self.datasets.values()])

        # client local dataloaders
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device

        self.dataloaders = {}
        from torch.utils.data import DataLoader
        for key, dataset in self.datasets.items():
            if key in ['train', ]:
                # for training set, we shuffle the data
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                   num_workers=self.num_workers)
            elif key in ['valid', 'test', ]:
                # for testing set, it is not necessary to shuffle
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                   num_workers=self.num_workers)

    def local_train(self, model, loss_func, metric_func, optimizer, num_iters, dataset, domain_id):
        """
        The server send the {model}, {loss_func}, and {optimizer} to client.
        The client will train the {model} with given configureation for {num_iters} rounds.
        """

        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        iterator = iter(self.dataloaders[dataset])
        for it in range(num_iters):

            # Get a batch of data (may iterate the dataset for multiple rounds)
            try:
                *X, Y = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloaders[dataset])  # reset iterator, dataset is shuffled
                *X, Y = next(iterator)

            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)
            D = torch.ones_like(Y) * int(domain_id)  # domain id

            # get prediction
            logits = model(*X, Y)

            loss = loss_func(logits, D)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples

                total_loss += loss.item() * num_examples

                metric = metric_func(logits, D)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric

    def local_eval(self, model, loss_func, metric_func, dataset, domain_id):
        """
        The server send the {model}, {loss_func} to client.
        The client test the {model} with its local train/val/test dataset.
        """
        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        with torch.no_grad():
            for *X, Y in self.dataloaders[dataset]:
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)
                D = torch.ones_like(Y) * int(domain_id)  # domain id

                # get prediction
                logits = model(*X, Y)
                loss = loss_func(logits, D)
                metric = metric_func(logits, D)

                num_examples = len(X[0])
                total_examples += num_examples

                total_loss += loss.item() * num_examples

                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric
