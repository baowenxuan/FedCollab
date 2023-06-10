import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def load_rotation_config(rotation_config):
    name, *params = rotation_config.split('_')
    if name == '20client':
        angle = int(params[0])
        config = {}
        for cid in range(0, 5):
            config[cid] = - angle // 2
        for cid in range(5, 10):
            config[cid] = + angle // 2
        for cid in range(10, 15):
            config[cid] = 180 - angle // 2
        for cid in range(15, 20):
            config[cid] = 180 + angle // 2

    else:
        raise NotImplementedError

    return config


def my_rotate(img, angle=0, padding_mode='symmetric'):
    """
    torchvision.transforms.functional.rotate use zero padding, which might be too obvious for a domain classifier.
    Therefore, we implement rotate with symmetric padding.
    """
    size = img.shape[-2:]
    img = img.cuda()
    img = TF.pad(img=img, padding=size[0] // 2, padding_mode=padding_mode)
    img = TF.rotate(img=img, angle=angle)
    img = TF.center_crop(img=img, output_size=size)
    img = img.cpu()
    return img


class RotatedDataset(Dataset):
    def __init__(self, dataset, angle=0, padding_mode='symmetric'):
        self.dataset = dataset
        self.angle = angle
        self.padding_mode = padding_mode

    def __getitem__(self, item):
        img, target = self.dataset[item]
        img = my_rotate(img, self.angle, self.padding_mode)
        return img, target

    def __len__(self):
        return len(self.dataset)
