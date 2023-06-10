import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


def create_loss(name='ce'):
    """
    loss function must be differentiable
    """
    if name == 'ce':
        return F.cross_entropy
    elif name == 'bce':
        return lambda output, target: F.binary_cross_entropy_with_logits(output.view(-1), target.view(-1).float())
    elif name == 'kl_div':
        return lambda pred_logits, target_logits: F.kl_div(F.log_softmax(pred_logits, dim=1), F.softmax(target_logits))
    elif name == 'mutual_kl_div':
        return lambda logits1, logits2: 0.5 * F.kl_div(F.log_softmax(logits1, dim=1), F.softmax(logits2)) + \
                                        0.5 * F.kl_div(F.log_softmax(logits2, dim=1), F.softmax(logits1))
    elif name == 'mean':
        return lambda logits, domain_id: - torch.mean((domain_id * 2 - 1).view(-1, 1) * logits)
    else:
        raise NotImplementedError('Unknown loss name: %s' % name)
