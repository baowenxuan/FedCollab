import torch
import os
from tqdm import tqdm

from dataset import create_fed_dataset
from utils import pickle_load, pickle_save
from options import args_parser


def main(args):
    train_datasets = create_fed_dataset(args)

    # pre-save it to a tensor
    train_tensor = {}
    for cid in tqdm(train_datasets.keys()):
        train_tensor[cid] = {}

        datasets = train_datasets[cid]

        for part in ['train', 'test']:
            train_tensor[cid][part] = {}

            dataset = datasets[part]

            X = []
            Y = []
            for i in range(len(dataset)):
                x, y = dataset[i]
                X.append(x)
                Y.append(y)

            X = torch.stack(X).cpu()
            Y = torch.LongTensor(Y).cpu()
            train_tensor[cid][part]['X'] = X
            train_tensor[cid][part]['Y'] = Y

    path = os.path.join(args.data_dir, 'rotated-cifar10.pkl')
    pickle_save(train_tensor, path, 'wb')

    print('Rotated dataset is saved to:', path)


if __name__ == '__main__':
    args = args_parser()
    print('Processing rotation')
    main(args)
