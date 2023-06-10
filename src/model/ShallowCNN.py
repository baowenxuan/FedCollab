import torch.nn as nn

from .Model import Model


class ShallowCNN(Model):
    """
    Shallow CNN, minimum input width and height is 18.
    """
    def __init__(self, shape_in, shape_out):
        super(ShallowCNN, self).__init__()

        in_channels = shape_in[0]
        h = ((((shape_in[1] - 2) // 2) - 2) // 2) - 2
        w = ((((shape_in[2] - 2) // 2) - 2) // 2) - 2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=h * w * 64, out_features=64)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(in_features=64, out_features=shape_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(x.shape[0], -1)  # (num_samples, num_features)
        x = self.linear4(x)
        x = self.relu4(x)

        x = self.linear5(x)

        return x


def test():
    import torch
    # x = torch.randn([1, 3, 2, 2])
    # conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
    # print(conv(x).shape)

    model = ShallowCNN(shape_in=(3, 18, 29), shape_out=10)
    x = torch.randn(2, 3, 18, 29)
    print(model(x).shape)
