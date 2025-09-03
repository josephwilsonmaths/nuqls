from models import *
import utils.training
from torchvision.models import resnet50, ResNet50_Weights
from posteriors.be.wide_resnet_batchensemble import Wide_ResNet_BatchEnsemble

def get_model(name, n_outputs, n_channels):
    if name == 'lenet':
        network = LeNet5()
    if name == 'lenet_custom':
        network = LeNet5_custom()
    elif name == 'resnet9':
        network = ResNet9(in_channels=n_channels, num_classes=n_outputs)
    elif name == 'wrn':
        network = WRN(depth=28, widening_factor=5, num_classes = n_outputs)
    elif name == 'resnet50':
        network = ResNet50(in_channels=n_channels, num_classes = n_outputs)
    elif name == 'smallwrn':
        wrn_n = 1
        network = Wide_ResNet_BatchEnsemble(channels=n_channels,
                                    depth=6*wrn_n+4,
                                    widen_factor=1,
                                    dropout_rate=0,
                                    num_classes=n_outputs,
                                    num_models=1,
                                    pool_number=7)
    elif name == 'resnet50_imagenet':
        network = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Apply standard initialisation - maybe change to have different initialization options, ntk etc.
    network.apply(utils.training.init_weights)

    return network



                