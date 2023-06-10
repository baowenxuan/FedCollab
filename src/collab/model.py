import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

from math import prod


class LinearDivergenceEstimator(nn.Module):
    """
    A Linear divergence estimator
    It cannot handle concept shift
    """

    def __init__(self, feature_dim, label_dim=None):
        super(LinearDivergenceEstimator, self).__init__()

        self.use_label = (label_dim is not None)  # by default no label

        # feature
        self.flatten = nn.Flatten()
        self.linear1_feat = nn.Linear(in_features=prod(feature_dim), out_features=1)

        if self.use_label:
            self.label_dim = label_dim
            self.linear1_label = nn.Linear(in_features=label_dim, out_features=1)

    def forward(self, x, y=None):
        # feature
        x = self.flatten(x)
        x = self.linear1_feat(x)

        if self.use_label:
            y = F.one_hot(y, num_classes=self.label_dim).float()
            y = self.linear1_label(y)
            x = x + y

        return x


class MLPDivergenceEstimator(nn.Module):
    def __init__(self, feature_dim, label_dim=None, c=0.01):
        super(MLPDivergenceEstimator, self).__init__()

        self.use_label = (label_dim is not None)  # by default no label

        self.c = c

        # feature
        self.flatten = nn.Flatten()
        self.linear1_feat = nn.Linear(in_features=prod(feature_dim), out_features=200)

        if self.use_label:
            self.label_dim = label_dim
            self.linear1_label = nn.Linear(in_features=label_dim, out_features=200)

        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(200, 1)

    def forward(self, x, y=None):
        # feature
        x = self.flatten(x)
        x = self.linear1_feat(x)

        # label
        if self.use_label:
            nu = self.label_dim / ((self.label_dim - 1) ** 0.5)  # to normalize the scale of x and y input
            y = nu * F.one_hot(y, num_classes=self.label_dim).float()
            # label information is more important than feature
            y = self.linear1_label(y)
            x = x + 2 * y

        x = self.relu1(x)

        x = self.linear2(x)

        return x

    def clip(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-self.c, self.c)

    def normalize(self):
        with torch.no_grad():
            if self.use_label:
                raise NotImplementedError

            else:
                norm1 = torch.linalg.norm(self.linear1_feat.weight, ord=2)
                norm2 = torch.linalg.norm(self.linear2.weight, ord=2)
                self.linear1_feat.weight /= norm1
                self.linear1_feat.bias /= norm1
                self.linear2.weight /= norm2
                self.linear2.bias /= (norm1 * norm2)


class MLPDivergenceEstimatorWithFeatureExtractor(nn.Module):
    def __init__(self, feature_dim, label_dim=None, c=0.01, freeze_extractor=True):
        super(MLPDivergenceEstimatorWithFeatureExtractor, self).__init__()

        self.use_label = (label_dim is not None)  # by default no label

        self.c = c

        self.freeze = freeze_extractor

        # feature
        self.feature_extractor = resnet18(pretrained=True)
        del self.feature_extractor.fc
        self.feature_extractor.fc = nn.Linear(512, 200)

        if self.freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.feature_extractor.fc.parameters():
                param.requires_grad = True

        # self.flatten = nn.Flatten()
        # self.linear1_feat = nn.Linear(in_features=prod(feature_dim), out_features=200)

        if self.use_label:
            self.label_dim = label_dim
            self.linear1_label = nn.Linear(in_features=label_dim, out_features=200)

        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(200, 1)

    def forward(self, x, y=None):
        # feature
        # x = self.flatten(x)
        # x = self.linear1_feat(x)
        x = self.feature_extractor(x)

        # label
        if self.use_label:
            nu = self.label_dim / ((self.label_dim - 1) ** 0.5)  # to normalize the scale of x and y input
            y = nu * F.one_hot(y, num_classes=self.label_dim).float()
            # label information is more important than feature
            y = self.linear1_label(y)
            x = x + 2 * y

        x = self.relu1(x)

        x = self.linear2(x)

        return x

    def train(self, mode=True):
        super(MLPDivergenceEstimatorWithFeatureExtractor, self).train(mode)
        if self.freeze:
            self.feature_extractor.eval()
            self.feature_extractor.fc.train()


def test():
    X = torch.cat([torch.Tensor([1, 0]).repeat(10, 1), torch.Tensor([0, 1]).repeat(10, 1)])
    Y = torch.cat([torch.zeros(5), torch.ones(10), torch.zeros(5)]).long()
    Z = torch.cat([torch.zeros(5), torch.ones(5), torch.zeros(5), torch.ones(5)]).long()

    model = MLPDivergenceEstimator((2,), 2, 0.03)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    # loss_func = lambda output, target: F.binary_cross_entropy_with_logits(output.view(-1), target.view(-1).float())
    loss_func = lambda logits, domain_id: - torch.mean((domain_id * 2 - 1).view(-1, 1) * logits)

    losses = []
    for i in range(100):
        optimizer.zero_grad()
        pred = model(X, Y)
        loss = loss_func(pred, Z)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            model.clip()

    print(losses)
