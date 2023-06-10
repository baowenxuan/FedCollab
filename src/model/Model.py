import torch
import torch.nn as nn
from collections import OrderedDict


class Model(nn.Module):
    """
    Model base class
    """

    def forward(self, x):
        """
        Output = Input
        :param x:
        :return:
        """
        return x

    def uploaded_state_dict(self):
        """
        Parameters that are uploaded to the server for aggregating
        By default, it is all the parameters
        :return:
        """
        return self.state_dict()

    def personal_state_dict(self):
        """
        Parameters that are personal:
        (1) not uploaded to the server, but
        (2) not freezed, is updated during local training
        E.g., batch-norm layers, personalized layers
        By default, it is empty
        :return:
        """
        return OrderedDict()

    def freezed_state_dict(self):
        """
        Parameters that are freezed, not for training. E.g., word embeddings for federated fine-tuning.
        By default, it is empty
        :return:
        """
        return OrderedDict()

    def updated_state_dict(self):
        """
        Parameters that are not freezed, which is either uploaded for aggregation or personal.
        :return:
        """
        return self.state_dict()

    def trainable_parameter_tensor(self):
        trainable_parameters = [tensor.view(-1) for tensor in self.parameters() if tensor.requires_grad]
        return torch.cat(trainable_parameters)


def test():
    model = Model()
    print(model.state_dict())  # OrderedDict()
    print(model.uploaded_state_dict())  # OrderedDict()
    print(model.personal_state_dict())  # OrderedDict()
    print(model.freezed_state_dict())  # OrderedDict()
