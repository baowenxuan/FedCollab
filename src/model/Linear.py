import torch
import torch.nn as nn
from math import prod

from .Model import Model


class Linear(Model):
    """
    Simplest linear model for linear regression and logistic regression
    """

    def __init__(self, shape_in, shape_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=prod(shape_in), out_features=shape_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # (num_samples, num_features)
        x = self.linear(x)
        return x


def test():
    model = Linear(shape_in=(1, 28, 28), shape_out=10)
    X = torch.randn(5, 1, 28, 28)
    logits = model(X)
    print(logits.shape)

    print(model.state_dict().values())
    print(X.numel())

