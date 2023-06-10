import torch
from torch.utils.data import Dataset


def load_labelswap_config(labelswap_config, num_labels):
    name, *params = labelswap_config.split('_')
    if name == '20client':
        num_swaps = int(params[0])
        map1 = {i: i for i in range(num_labels)}
        map2 = {**{i: num_swaps - 1 - i for i in range(num_swaps)},
                **{i: i for i in range(num_swaps, num_labels)}}

        map3 = {i: num_labels - 1 - i for i in range(num_labels)}
        map4 = {**{i: num_labels - num_swaps + i for i in range(num_swaps)},
                **{i: num_labels - 1 - i for i in range(num_swaps, num_labels)}}

        config = {}
        for cid in range(0, 5):
            config[cid] = map1
        for cid in range(5, 10):
            config[cid] = map2
        for cid in range(10, 15):
            config[cid] = map3
        for cid in range(15, 20):
            config[cid] = map4

        print(map2, map3, map4)

    else:
        raise NotImplementedError

    return config


class LabelSwapDataset(Dataset):
    def __init__(self, dataset, label_mapper):
        self.dataset = dataset
        self.label_mapper = label_mapper

    def __getitem__(self, item):
        img, target = self.dataset[item]
        target = self.label_mapper[target]
        return img, target

    def __len__(self):
        return len(self.dataset)
