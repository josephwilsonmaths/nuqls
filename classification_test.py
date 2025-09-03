import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
import tqdm
import models as model
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import utils.training

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

training_data = datasets.MNIST(
    root="data/MNIST",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data/MNIST",
    train=False,
    download=True,
    transform=ToTensor()
)

ood_test_data = datasets.FashionMNIST(
    root="data/FashionMNIST",
    train=False,
    download=True,
    transform=ToTensor()
)

n_output = 10
n_channels = 1

map_net = model.LeNet5()

map_net.load_state_dict(torch.load(f'models/lenet_trained_mnist.pt', weights_only=True, map_location=torch.device('cpu')))
map_net.eval()

train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=100)
ood_test_dataloader = DataLoader(ood_test_data, batch_size=100)


from nuqls.posterior import Nuqls

nuqls_posterior = Nuqls(map_net, task='classification')
nuqls_posterior.pre_load(f'models/nuqls_lenet_mnist.pt')


prob_var_dict = {}


correct_var = []
incorrect_var = []
for x,y in test_dataloader:
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
for x,y in ood_test_dataloader:
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
for x,y in test_dataloader:
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
for x,y in ood_test_dataloader:
    sample = (torch.randn((S,len(y),10)) * scale_n).softmax(2)

    max_index = sample.mean(0).argmax(1)
    po_var = sample[:,range(len(max_index)),max_index].var(0)

    ood_var.append(po_var)
    
ood_var = torch.cat(ood_var)

prob_var_dict['BASE'] = {'id_correct': correct_var,
                    'id_incorrect': incorrect_var,
                    'ood': ood_var}    

import utils.metrics as metrics
metrics.plot_vmsp(prob_dict=prob_var_dict,
                          title=f'mnist',
                          save_fig='test.pdf')

torch.save(prob_var_dict, 'pvd.pt')