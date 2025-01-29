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
import utils.datasets as ds
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

parser = argparse.ArgumentParser(description='Classification Experiment')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--model', default='lenet', type=str, help='model: lenet, resnet9, resnet50')
parser.add_argument('--subsample', action='store_true', help='Use less datapoints for train and test.')
parser.add_argument('--verbose', action='store_true',help='verbose flag for all methods')
parser.add_argument('--save_var', action='store_true', help='save variances if on (memory consumption)')
parser.add_argument('--pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--progress', action='store_false')
args = parser.parse_args()

#--- Get hyperparameters from config file
config = configparser.ConfigParser()
config.read('utils/classification.ini')
field = f'{args.model}_{args.dataset}'
n_experiment = config.getint(field,'n_experiment')
epochs = config.getint(field,'epochs')
lr = config.getfloat(field,'lr')
wd = config.getfloat(field,'wd')
bs = config.getint(field,'bs')
S = config.getint(field,'S')
nuqls_S = config.getint(field,'nuqls_S')
nuqls_epoch = config.getint(field,'nuqls_epoch')
nuqls_lr = config.getfloat(field,'nuqls_lr')
nuqls_wd = config.getfloat(field,'nuqls_wd')
nuqls_bs = config.getint(field,'nuqls_bs')
nuqls_gamma = config.getfloat(field,'nuqls_gamma')
nuqls_parallel = config.getboolean(field,'nuqls_parallel')
print(f'nuqls_parallel = {nuqls_parallel}')
# ---





if args.dataset=='mnist':
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

    aucroc = {'fmnist': [],
              'kmnist': [],
              'rot_mnist': []}
if args.dataset=='fmnist':
    training_data = datasets.FashionMNIST(
        root="data/FashionMNIST",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data/FashionMNIST",
        train=False,
        download=True,
        transform=ToTensor()
    )

    aucroc = {'mnist': [],
              'rot_fmnist': []}
    
train_dataloader = DataLoader(training_data, batch_size=bs)
test_dataloader = DataLoader(test_data, batch_size=bs)
    
loss_fn = nn.CrossEntropyLoss()

# Setup directories
res_dir = f"./results/slu_comparison/{args.dataset}_{args.model}/"

ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
res_dir = res_dir + f"_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

for ei in tqdm.trange(n_experiment):
    print("\n--- experiment {} ---".format(ei))

    # MAP
    if args.model == 'mlp':
        map_net = model.MLP()
    if args.model == 'lenet_custom':
        map_net = model.LeNet5_custom()

    map_net.apply(utils.training.init_weights_he)
    map_net.eval()
    num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
    print(f"num weights = {num_weights}")
    map_net.to(device)

    if args.pre_load:
        print('Using pre-loaded weights!')
        map_net.load_state_dict(torch.load(f'models/{args.model}_trained_{args.dataset}.pt', weights_only=True, map_location=torch.device('cpu')))
        map_net.eval()
        map_net.to(device)
    else:
        optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
        scheduler = None

        train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                    model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                    scheduler=scheduler,epochs=epochs,
                                                                    verbose=args.verbose, progress_bar=args.progress)
        print("\nTrain Results:")
        print(f"Train Loss: {train_loss:.3f}; Train Acc: {train_acc:.1%}")
        torch.save(map_net.state_dict(), f'models/{args.model}_trained_{args.dataset}.pt')

    # NUQLS
    
    nuql = nuqls.classification_parallel_i(map_net)
    loss = nuql.train(train=training_data, train_bs = nuqls_bs, n_output=10, S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9)
    nuqls_predictions = nuql.test(test=test_data, test_bs=nuqls_bs)
    id_logit_var = nuqls_predictions.var(0)
    
    if args.dataset == 'mnist':
        if args.verbose:
            print('Computing FashionMNIST')
        ood_test_data = datasets.FashionMNIST(
            root="data/FashionMNIST",
            train=False,
            download=True,
            transform=ToTensor()
        )
        ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
        ood_logit_var = ood_nuqls_predictions.var(0)
        aucroc['fmnist'].append(metrics.ood_auc(id_logit_var.sum(1).detach().cpu(), ood_logit_var.sum(1).detach().cpu()))

        if args.verbose:
            print('Computing KMNIST')
        ood_test_data = datasets.KMNIST(
            root="data/KMNIST",
            train=False,
            download=True,
            transform=ToTensor()
        )
        ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
        ood_logit_var = ood_nuqls_predictions.var(0)
        aucroc['kmnist'].append(metrics.ood_auc(id_logit_var.sum(1).detach().cpu(), ood_logit_var.sum(1).detach().cpu()))

        # Rotated FMNIST
        aucroc['rot_mnist'].append(0)
        angles = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        for angle in angles:
            # MNIST
            if args.verbose:
                print(f'Computing Rotated ({angle}) MNIST')
            ood_test_data = ds.get_rotated_MNIST(angle)
            ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
            ood_logit_var = ood_nuqls_predictions.var(0)
            aucroc['rot_mnist'][ei] += metrics.ood_auc(id_logit_var.sum(1).detach().cpu(), ood_logit_var.sum(1).detach().cpu())
        aucroc['rot_mnist'][ei] /= len(angles)
        
    if args.dataset == 'fmnist':
        # MNIST
        if args.verbose:
            print('Computing MNIST')
        ood_test_data = datasets.MNIST(
            root="data/MNIST",
            train=False,
            download=True,
            transform=ToTensor()
        )
        ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
        ood_logit_var = ood_nuqls_predictions.var(0)
        aucroc['mnist'].append(metrics.ood_auc(id_logit_var.sum(1).detach().cpu(), ood_logit_var.sum(1).detach().cpu()))

        # Rotated FMNIST
        aucroc['rot_fmnist'].append(0)
        angles = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        for angle in angles:
            # MNIST
            if args.verbose:
                print(f'Computing Rotated ({angle}) FMNIST')
            ood_test_data = ds.get_rotated_FMNIST(angle)
            ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
            ood_logit_var = ood_nuqls_predictions.var(0)
            aucroc['rot_fmnist'][ei] += metrics.ood_auc(id_logit_var.sum(1).detach().cpu(), ood_logit_var.sum(1).detach().cpu())
        aucroc['rot_fmnist'][ei] /= len(angles)

    print("\nTest Prediction:")
    for d in aucroc.keys():
        print(f"{d}: AUCROC: {aucroc[d][ei]:.1%}\n")

## Record results
res_text = res_dir + f"result.txt"
results = open(res_text,'w')

percentage_metrics = [aucroc]

results.write(" --- MAP Training Details --- \n")
results.write(f"epochs: {epochs}; M: {S}; lr: {lr}; weight_decay: {wd}\n")

results.write("\n --- NUQLS Details --- \n")
results.write(f"epochs: {nuqls_epoch}; S: {nuqls_S}; lr: {nuqls_lr}; weight_decay: {nuqls_wd}; init scale: {nuqls_gamma}\n")

results.write("\n - NUQLS Result: - \n")
for d in aucroc.keys():
    results.write(f"{d}: AUCROC: {np.mean(aucroc[d]):.1%} +- {np.std(aucroc[d]):.1%}\n")

results.close()