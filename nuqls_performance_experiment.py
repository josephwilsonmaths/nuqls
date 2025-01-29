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
parser.add_argument('--progress', action='store_false')
parser.add_argument('--constant_loss', action='store_true')
args = parser.parse_args()

# Get dataset
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

    ood_test_data = datasets.FashionMNIST(
        root="data/FashionMNIST",
        train=False,
        download=True,
        transform=ToTensor()
    )

    training_data, val_data = torch.utils.data.random_split(training_data,[50000,10000])
    n_output = 10
    n_channels = 1

elif args.dataset=='fmnist':
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

    ood_test_data = datasets.MNIST(
        root="data/MNIST",
        train=False,
        download=True,
        transform=ToTensor()
    ) 

    training_data, val_data = torch.utils.data.random_split(training_data,[50000,10000])
    n_output = 10
    n_channels = 1

elif args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_data = datasets.CIFAR10(
        root="data/CIFAR10",
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = datasets.CIFAR10(
        root="data/CIFAR10",
        train=False,
        download=True,
        transform=transform_test
    )

    ood_test_data = datasets.CIFAR100(
        root="data/CIFAR100",
        train=False,
        download=True,
        transform=transform_test
    ) 

    # train_data, val_data = torch.utils.data.random_split(training_data,[50000,10000]) -> results in performance drop for CIFAR
    val_data = test_data
    n_output = 10
    n_channels = 3

elif args.dataset=='cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_data = datasets.CIFAR100(
        root="data/CIFAR100",
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = datasets.CIFAR100(
        root="data/CIFAR100",
        train=False,
        download=True,
        transform=transform_test
    )

    ood_test_data = datasets.CIFAR10(
        root="data/CIFAR10",
        train=False,
        download=True,
        transform=transform_test
    ) 

    # train_data, val_data = torch.utils.data.random_split(training_data,[50000,10000]) -> results in performance drop for CIFAR
    val_data = test_data
    n_output = 100
    n_channels = 3

if args.subsample:
    N_TRAIN = 1000
    N_TEST = 11
    training_data = torch.utils.data.Subset(training_data,range(N_TRAIN))
    test_data = torch.utils.data.Subset(test_data,range(N_TEST))
    val_data = torch.utils.data.Subset(val_data,range(N_TEST))
    ood_test_data = torch.utils.data.Subset(ood_test_data,range(N_TEST))

metric = MulticlassCalibrationError(num_classes=n_output,n_bins=10,norm='l1')
full_train_dataloader = DataLoader(training_data, len(training_data))
print(len(training_data))
train_x,train_y = next(iter(full_train_dataloader))

full_test_dataloader = DataLoader(test_data, len(test_data))
test_x,test_y = next(iter(full_test_dataloader))

full_ood_dataloader = DataLoader(ood_test_data, len(ood_test_data))
ood_test_x,ood_test_y = next(iter(full_ood_dataloader))

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
nuqls_wd = config.getfloat(field,'nuqls_wd')
nuqls_parallel = config.getboolean(field,'nuqls_parallel')
print(f'nuqls_parallel = {nuqls_parallel}')
# ---

train_dataloader = DataLoader(training_data, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=bs)
ood_test_dataloader = DataLoader(ood_test_data, batch_size=bs)

loss_fn = nn.CrossEntropyLoss()

# Setup directories
if args.subsample:
    res_dir = f"./results/nuqls_performance_experiment/{args.constant_loss}/subsample/{args.dataset}_{args.model}/"
else:
    res_dir = f"./results/nuqls_performance_experiment/{args.constant_loss}/{args.dataset}_{args.model}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

for ei in tqdm.trange(n_experiment):

    dict = {'map_train_loss': 0,
            'lr': [],
            'epoch': [],
            'gamma': [],
            'bs': [],
            'nuqls_train_loss': [],
            'acc': [],
            'ece': [],
            'aucroc': [],
            'oodauc': [],}

    print("\n--- experiment {} ---".format(ei))
    ### --------- MAP --------- ###
    if args.model == 'lenet':
        map_net = model.LeNet5()
    elif args.model == 'resnet9':
        map_net = model.ResNet9(in_channels=n_channels, num_classes=n_output)
    elif args.model == 'wrn':
        map_net = model.WRN(depth=28, widening_factor=5, num_classes = n_output)
    elif args.model == 'resnet50':
        map_net = model.ResNet50(in_channels=n_channels, num_classes = n_output)
    map_net.apply(utils.training.init_weights)
    map_net.eval()
    num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
    print(f"num weights = {num_weights}")

    # Multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        map_net = nn.DataParallel(map_net)
    
    map_net.to(device)

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    elif args.dataset == 'cifar10':
        if args.model == 'resnet9':
            optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
            scheduler = None
        elif args.model == 'resnet50':
            optimizer = torch.optim.SGD(map_net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    elif args.dataset == 'cifar100':
        optimizer = torch.optim.SGD(map_net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

    train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                scheduler=scheduler,epochs=epochs,
                                                                verbose=args.verbose, progress_bar=args.progress)
    
    dict['map_train_loss'] = train_loss

    ### --------- NUQLS --------- ###

    # dict = {'map_train_loss': 0,
    #         'lr': [],
    #         'epoch': [],
    #         'gamma': [],
    #         'bs': [],
    #         'nuqls_train_loss': 0,
    #         'acc': 0,
    #         'ece': 0,
    #         'aucroc': 0,
    #         'oodauc': 0,}

    if not args.constant_loss:
        learning_rate_list = [1e-4, 1e-2]
        epochs_list = [10,50]
        gamma_list = [0.01,0.05,0.1,0.5,1,5]
        batch_size_list = [152,10000]

        for nuqls_bs in batch_size_list:
            for nuqls_lr in learning_rate_list:            
                for nuqls_epoch in epochs_list:
                    for nuqls_gamma in gamma_list:

                        print(f'\nbs: {nuqls_bs}, lr: {nuqls_lr}, gamma: {nuqls_gamma}, epochs: {nuqls_epoch}')

                        dict['lr'].append(nuqls_lr)
                        dict['epoch'].append(nuqls_epoch)
                        dict['gamma'].append(nuqls_gamma)
                        dict['bs'].append(nuqls_bs)

                        # nuqls_predictions, ood_nuqls_predictions, loss = nuqls.classification_parallel_i(net = map_net, train=training_data, test=test_data, 
                        #                                             ood_test=ood_test_data, train_bs = nuqls_bs, test_bs = nuqls_bs, 
                        #                                             n_output=n_output, S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9)
                        
                        nuql = nuqls.classification_parallel_i(map_net)
                        loss = nuql.train(train=training_data, train_bs = nuqls_bs, n_output=n_output, S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9)
                        nuqls_predictions = nuql.test(test=test_data, test_bs=nuqls_bs)
                        ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
                        
                        dict['nuqls_train_loss'].append(loss)

                        id_predictions = nuqls_predictions.softmax(dim=2)
                        ood_predictions = ood_nuqls_predictions.softmax(dim=2)
                        id_mean = nuqls_predictions.softmax(dim=2).mean(dim=0)
                        ood_mean = ood_nuqls_predictions.softmax(dim=2).mean(dim=0)

                        p_dist = Categorical(probs=id_mean.to(device))
                        dict['acc'].append((id_mean.argmax(1).to(device) == test_y.to(device)).type(torch.float).mean().item())
                        dict['ece'].append(metric(id_mean.to(device),test_y.to(device)).cpu().item())
                        oodauc, aucroc = metrics.auc_metric(id_mean, ood_mean, logits=False)
                        dict['aucroc'].append(aucroc)
                        dict['oodauc'].append(oodauc)
    else:
        nuqls_lr = 1e-2
        nuqls_epoch = 1000
        loss_tol = train_loss
        gamma_list = torch.logspace(-3,0.5,10)
        nuqls_bs = 152

        print(f'loss tolerance = {loss_tol}')

        for nuqls_gamma in gamma_list:

            print(f'\ngamma: {nuqls_gamma}')

            dict['lr'].append(nuqls_lr)
            dict['epoch'].append(nuqls_epoch)
            dict['gamma'].append(nuqls_gamma)
            dict['bs'].append(nuqls_bs)

            nuql = nuqls.classification_parallel_i(map_net)
            loss = nuql.train(train=training_data, train_bs = nuqls_bs, n_output=n_output, S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9, loss_tol=loss_tol, verbose=True)
            nuqls_predictions = nuql.test(test=test_data, test_bs=nuqls_bs)
            ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)
            
            dict['nuqls_train_loss'].append(loss)

            id_predictions = nuqls_predictions.softmax(dim=2)
            ood_predictions = ood_nuqls_predictions.softmax(dim=2)
            id_mean = nuqls_predictions.softmax(dim=2).mean(dim=0)
            ood_mean = ood_nuqls_predictions.softmax(dim=2).mean(dim=0)

            p_dist = Categorical(probs=id_mean.to(device))
            dict['acc'].append((id_mean.argmax(1).to(device) == test_y.to(device)).type(torch.float).mean().item())
            dict['ece'].append(metric(id_mean.to(device),test_y.to(device)).cpu().item())
            oodauc, aucroc = metrics.auc_metric(id_mean, ood_mean, logits=False)
            dict['aucroc'].append(aucroc)
            dict['oodauc'].append(oodauc)        
    
    # Save dict
    torch.save(dict,res_dir + f'dict_{ei}.pt')