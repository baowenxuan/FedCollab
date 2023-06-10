import os
import json
from tqdm import tqdm


def load_data(dataset_name='mnist', split='train', data_dir='~/data', first_k=None):
    users = []
    num_samples = []
    user_data = {}

    data_dir = os.path.expanduser(data_dir)
    data_dir = os.path.join(data_dir, 'leaf', 'data', dataset_name, 'data')  # data directory
    subdir = os.path.join(data_dir, split)

    files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]

    if first_k is not None:
        files = files[:first_k]

    # print(files)
    for f in tqdm(files):
        file_dir = os.path.join(subdir, f)

        with open(file_dir) as inf:
            data = json.load(inf)

        users.extend(data['users'])
        num_samples.extend(data['num_samples'])
        user_data.update(data['user_data'])

    return users, num_samples, user_data