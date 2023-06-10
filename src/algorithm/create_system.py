# global federated learning
from .FedAvg import FedAvgServer
from .FedProx import FedProxServer
from .FedNova import FedNovaServer

# personalized federated learning
from .LocalFinetune import LocalFinetuneServer
from .PerFedAvg import PerFedAvgServer
from .pFedMe import pFedMeServer
from .Ditto import DittoServer


def create_system(algorithm, client_datasets, args):
    if algorithm == 'fedavg':
        server = FedAvgServer(client_datasets, args)
    elif algorithm == 'fedprox':
        server = FedProxServer(client_datasets, args)
    elif algorithm == 'fednova':
        server = FedNovaServer(client_datasets, args)

    elif algorithm == 'finetune':
        server = LocalFinetuneServer(client_datasets, args)
    elif algorithm == 'perfedavg':
        server = PerFedAvgServer(client_datasets, args)
    elif algorithm == 'pfedme':
        server = pFedMeServer(client_datasets, args)
    elif algorithm == 'ditto':
        server = DittoServer(client_datasets, args)

    else:
        raise NotImplementedError('Unknown Federated Learning Algorithm. ')

    return server
