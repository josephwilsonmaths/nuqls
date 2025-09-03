import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
import tqdm
import numpy as np
import models as model
from torch import nn
from torch.utils.data import DataLoader
from laplace import Laplace
import os
import posteriors.util as pu
import posteriors.swag as swag
import posteriors.nuqls as nuqls
import posteriors.lla_s as lla_s
from posteriors.lla.likelihoods import Categorical
import utils.metrics as metrics
from posteriors.valla.utils.metrics import SoftmaxClassification, OOD, psd_safe_cholesky
from posteriors.valla.src.valla import VaLLAMultiClassBackend
from posteriors.valla.src.utils import smooth
from posteriors.valla.utils.process_flags import manage_experiment_configuration
from posteriors.valla.utils.pytorch_learning import fit_map_crossentropy, fit, forward, score
import time
import datetime
from time import process_time as timer
from torchmetrics.classification import MulticlassCalibrationError
import argparse
import warnings
from scipy.cluster.vq import kmeans2
import configparser
from nuqls.posterior import Nuqls
from torchvision.models import resnet50, ResNet50_Weights
import utils.metrics as metrics

import utils.training

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n Using {device} device")
print(f"CUDA version: {torch.version.cuda}")

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)

# training_data = datasets.ImageNet(
#     root = '/QRISdata/Q7521/ImageNet',
#     split = 'train',
#     transform = preprocess
# )

training_data = datasets.ImageFolder(
    root='/scratch/licenseddata/imagenet/imagenet-1k/train',
    transform = preprocess
)

mean_test = [0.485, 0.456, 0.406]
std_test = [0.229, 0.224, 0.225]

test_transform = transforms.Compose(
    [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean_test, std_test)])

test_data = datasets.ImageFolder(
    root='/scratch/licenseddata/imagenet/imagenet-1k/val',
    transform = test_transform
)

mean_ood = [0.485, 0.456, 0.406]
std_ood = [0.229, 0.224, 0.225]

ood_transform = transforms.Compose(
    [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean_ood, std_ood)])

print('loading datasets')

ood_test_data = datasets.ImageFolder(r'data/imagenet-o/imagenet-o', transform=ood_transform)

n_output = 1000
n_channels = 3

print('loaded datasets')

map_net = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

nuqls_posterior = Nuqls(map_net, task='classification')
pre_load = f'models/nuqls_resnet50_imagenet.pt'
nuqls_posterior.pre_load(pre_load)

testloader = DataLoader(test_data, batch_size=100)
ood_testloader = DataLoader(test_data, batch_size=100)

prob_var_dict = {}

res_dir = f"./results/image_classification/imagenet_resnet50/"

correct_var = []
incorrect_var = []
for x,y in testloader:
    preds = nuqls_posterior.eval(x).softmax(dim=2)
    index = (preds.mean(0).argmax(1) == y)
    correct_preds = preds[:,index,:]
    incorrect_preds = preds[:,~index,:]

    max_index = correct_preds.mean(0).argmax(1)
    if sum(max_index) > 0:
        pi_correct_var = correct_preds[:,range(len(max_index)),max_index].var(0)
        correct_var.append(pi_correct_var)

    max_index = incorrect_preds.mean(0).argmax(1)
    if sum(max_index) > 0:
        pi_incorrect_var = incorrect_preds[:,range(len(max_index)),max_index].var(0)
        incorrect_var.append(pi_incorrect_var)

correct_var = torch.cat(correct_var)
incorrect_var = torch.cat(incorrect_var)

ood_var = []
for x,y in ood_testloader:
    preds = nuqls_posterior.eval(x).softmax(dim=2)

    max_index = preds.mean(0).argmax(1)
    po_var = preds[:,range(len(max_index)),max_index].var(0)

    ood_var.append(po_var)

ood_var = torch.cat(ood_var)

prob_var_dict['NUQLS'] = {'id_correct': correct_var,
                    'id_incorrect': incorrect_var,
                    'ood': ood_var}

S = 10
scale_n = 1

correct_var = []
incorrect_var = []
for x,y in testloader:
    sample = (torch.randn((S,len(y),10)) * scale_n).softmax(2)
    index = (sample.mean(0).argmax(1) == y)
    correct_preds = sample[:,index,:]
    incorrect_preds = sample[:,~index,:]
    
    max_index = correct_preds.mean(0).argmax(1)
    if sum(max_index) > 0:
        pi_correct_var = correct_preds[:,range(len(max_index)),max_index].var(0)
        correct_var.append(pi_correct_var)

    max_index = incorrect_preds.mean(0).argmax(1)
    if sum(max_index) > 0:
        pi_incorrect_var = incorrect_preds[:,range(len(max_index)),max_index].var(0)
        incorrect_var.append(pi_incorrect_var)

correct_var = torch.cat(correct_var)
incorrect_var = torch.cat(incorrect_var)

ood_var = []
for x,y in ood_testloader:
    sample = (torch.randn((S,len(y),10)) * scale_n).softmax(2)

    max_index = sample.mean(0).argmax(1)
    po_var = sample[:,range(len(max_index)),max_index].var(0)

    ood_var.append(po_var)
    
ood_var = torch.cat(ood_var)

prob_var_dict['BASE'] = {'id_correct': correct_var,
                    'id_incorrect': incorrect_var,
                    'ood': ood_var}    

torch.save(prob_var_dict, res_dir + 'pvd.pt')

metrics.plot_vmsp(prob_dict=prob_var_dict,
                          title=f'Imagenet ResNet50',
                          save_fig=res_dir + f"vmsp_plot.pdf")
