import torch.nn as nn
from torchvision.models import resnet18

from .Model import Model

class ResNet18(Model):
    """
    ResNet
    """
    def __init__(self, shape_out):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(pretrained=True)

        # replace the final fully connected layer
        del self.backbone.fc
        self.backbone.fc = nn.Linear(512, shape_out)


    def forward(self, x):
        if x.shape[1] == 1:  # 1-channel image
            x = x.repeat(1, 3, 1, 1)

        return self.backbone(x)


def test():
    model = ResNet18(shape_out=10)
    total_num = sum(p.numel() for name, p in model.state_dict().items())
    print(total_num)