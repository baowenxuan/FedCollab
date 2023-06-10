import os
import argparse
import numpy as np

from utils import pickle_load


def get_accs(data, model='GM'):
    key = '%s_test_metrics' % model

    mat = np.zeros((int(data['args'].partition_config.split('_')[1]), data['args'].gm_rounds))

    for coalition, history in zip(data['collab'], data['histories']):

        for cid, idx in history['cid2idx'].items():
            mat[cid] = np.array((np.array(history[key])[:, idx]))

    return mat[:, -1]


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--history_path', default='../history/default.pkl')

    parser.add_argument('--ref_history_path', default='../history/default.pkl')

    parser.add_argument('--ref', action='store_true', default=False)

    args = parser.parse_args()

    return args


def main(args):
    # load training history
    data = pickle_load(args.history_path, multiple=False)
    # extract accuracy for each client
    accs = get_accs(data, model='GM')  # change to 'PM' for personalized FL

    print('Acc: %.4f' % np.mean(accs))

    if args.ref:
        ref_data = pickle_load(args.ref_history_path, multiple=False)
        ref_accs = get_accs(ref_data, model='GM')

        print('IPR: %.4f' % np.mean((accs - ref_accs) > 0))
        print('RSD: %.4f' % np.std((accs - ref_accs)))


if __name__ == '__main__':
    args = args_parser()
    main(args)
