import segmentation_models_pytorch as smp
from utils.UNet import Unet


def get_model(num_classes=2, use_sigmoid=False, type='unet', in_channels=3):
    ENCODER = 'efficientnet-b2'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    ACTIVATION = 'sigmoid' if use_sigmoid else None
    aux_params = dict(
        pooling='max',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        activation='sigmoid',  # activation function, default is None
        classes=num_classes,  # define number of output labels
    )
    if type == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=num_classes,
            activation=ACTIVATION,
            aux_params=aux_params,
            in_channels=in_channels)
    else:
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=num_classes,
            activation=ACTIVATION,
            aux_params=aux_params,
            in_channels=in_channels)
    model.cuda(0)
    preproc = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model, preproc

def get_custom_model(num_classes=2, use_sigmoid=False):
    unet = Unet(in_channels=2, num_classes=num_classes, use_sigmoid=use_sigmoid)
    unet.cuda(0)
    return unet
