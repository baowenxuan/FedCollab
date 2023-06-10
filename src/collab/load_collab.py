from utils import pickle_load


def load_collab(args, train_datasets):
    if args.collab_config == 'global':
        collab = [[cid for cid in train_datasets]]
        print('Global Training Collaboration:', collab)

    elif args.collab_config == 'local':
        collab = [[cid, ] for cid in train_datasets]
        print('Local Training Collaboration:', collab)

    else:
        collab = pickle_load(args.collab_path)
        print('Load Collaboration:', collab)

    return collab
