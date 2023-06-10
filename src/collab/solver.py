import torch
from tqdm import tqdm

from .evaluate import evaluate_with_fixed_weight, reduction


def discrete_solver(disc, m, beta, C, init='local', max_iter=100, shuffle=True):
    log = []
    N = len(beta)  # num clients

    # initialize
    if init == 'local':
        choice = torch.arange(len(beta))
    else:
        choice = init
        # raise NotImplementedError('Unknown initialization. ')

    for it in tqdm(range(max_iter)):
        if shuffle:
            cids = torch.randperm(N)
        else:
            cids = torch.arange(N)

        earlystop = True

        for query_id in cids:
            # print('Now query ID:', query_id, choices)
            c_prev = choice[query_id].item()
            errors = evaluate_with_fixed_weight(choice, disc, m, beta, C)
            error_prev = reduction(errors, beta, typ='unweighted_average')
            log.append(error_prev.cpu().numpy())

            for c in range(N):
                choice[query_id] = c
                errors = evaluate_with_fixed_weight(choice, disc, m, beta, C)
                error = reduction(errors, beta, typ='unweighted_average')

                if error < error_prev:
                    c_prev = c
                    error_prev = error
                    earlystop = False

            choice[query_id] = c_prev

        if earlystop:
            break
    # print(choice)
    # choice = torch.LongTensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    errors = evaluate_with_fixed_weight(choice, disc, m, beta, C)
    error = reduction(errors, beta, typ='unweighted_average')

    return choice, error, log
