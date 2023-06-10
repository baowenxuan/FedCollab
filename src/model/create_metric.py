import torch


def create_metric(name='acc'):
    """
    metric function can be any function with scalar output.
    """
    if name == 'acc':
        return lambda logits, target: logits.argmax(dim=1).eq(target).float().mean()
    elif name == 'err':
        return lambda logits, target: logits.argmax(dim=1).ne(target).float().mean()
    elif name == 'bacc':
        return lambda logits, target: logits.ge(0).eq(target).float().mean()
    elif name == 'berr':
        return lambda logits, target: logits.ge(0).ne(target).float().mean()
    elif name == 'H_delta_H':
        return lambda logits1, logits2: logits1.argmax(dim=1).ne(logits2.argmax(dim=1)).float().mean()
    elif name == 'mean':
        return lambda logits, domain_id: torch.mean((domain_id * 2 - 1).view(-1, 1) * logits)
    else:
        raise NotImplementedError('Unknown metric name: %s' % name)