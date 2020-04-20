import segmentation_models_pytorch as smp


def get_model(num_classes=2, use_sigmoid=False):
    ENCODER = 'efficientnet-b0'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    ACTIVATION = 'sigmoid' if use_sigmoid else None
    aux_params = dict(
        pooling='max',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        activation='sigmoid',  # activation function, default is None
        classes=num_classes,  # define number of output labels
    )
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=num_classes,
        activation=ACTIVATION,
        aux_params=aux_params
    )
    model.cuda(0)
    preproc = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model, preproc
