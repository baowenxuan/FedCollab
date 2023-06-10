import torch
import torch.nn as nn
import numpy as np

from .Model import Model



class TwoNN(Model):
    """
    2NN, a multi-layer perceptron with 2 hidden layers
    Reference:

    """
    def __init__(self, shape_in=(1, 28, 28), shape_out=10):
        super(TwoNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(shape_in)), 200)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(200, shape_out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def test():
    model = TwoNN(shape_in=(1, 28, 28), shape_out=10)
    X = torch.randn(5, 1, 28, 28)
    logits = model(X)
    print(logits.shape)

    print(model.state_dict())
