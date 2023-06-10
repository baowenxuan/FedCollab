from .Linear import Linear
from .TwoNN import TwoNN
from .ShallowCNN import ShallowCNN
from .ResNet import ResNet18


def create_model(args):
    """
    Create model
    :param args:
    :return:
    """
    shape_in = args.shape_in
    shape_out = args.shape_out

    if args.model == 'linear':
        model = Linear(shape_in=shape_in, shape_out=shape_out)
    elif args.model == '2nn':
        model = TwoNN(shape_in=shape_in, shape_out=shape_out)
    elif args.model == 'cnn':
        model = ShallowCNN(shape_in=shape_in, shape_out=shape_out)
    elif args.model == 'resnet18':
        model = ResNet18(shape_out=shape_out)
    else:
        raise NotImplementedError('Unknown model. ')

    model.to(args.device)

    return model
