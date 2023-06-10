import torch

from .utils import choice_to_fixed_alpha


def evaluate_with_fixed_weight(choice, disc, m, beta, C):
    alpha = choice_to_fixed_alpha(choice, beta)
    errors = evaluate(alpha, disc, m, beta, C)
    return errors


def evaluate(alpha, disc, m, beta, C):
    gen_errors = torch.sqrt((torch.square(alpha) / beta).sum(dim=1) / m)
    shift_errors = (alpha * disc).sum(dim=1)

    errors = C * gen_errors + shift_errors
    return errors


def reduction(errors, beta, typ='weighted_average'):
    if typ == 'unweighted_average':
        return errors.mean()
    elif typ == 'weighted_average':
        return (errors * beta).sum()
    elif typ == 'log_sum_exp':
        return torch.log(torch.exp(errors).sum())


def test():
    alpha = torch.Tensor([
        [0.4, 0.6, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.7],
        [0.0, 0.0, 0.3, 0.7],
    ])

    disc = torch.Tensor([
        [0.0, 0.5, 1.0, 1.0],
        [0.5, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5, 0.0],
    ])

    m = torch.Tensor([20, ])

    beta = torch.Tensor([0.2, 0.3, 0.15, 0.35])

    C = 1

    choices = torch.LongTensor([0, 0, 1, 1])

    errors = evaluate(alpha, disc, m, beta, C)
    print(errors)

    errors2 = evaluate_with_fixed_weight(choices, disc, m, beta, C)
    print(errors2)

