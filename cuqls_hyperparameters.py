import torch
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
import utils.metrics as metrics
from posteriors.valla.utils.metrics import SoftmaxClassification, OOD, psd_safe_cholesky
from posteriors.valla.src.valla import VaLLAMultiClassBackend
from posteriors.valla.utils.pytorch_learning import fit
import time
import datetime
import argparse
import warnings
from scipy.cluster.vq import kmeans2
from nuqls.posterior import Nuqls
from cuqls.posterior import Cuqls
from posteriors.be.wide_resnet_batchensemble import Wide_ResNet_BatchEnsemble
import posteriors.be.util as be_util
import utils.training
import utils.classification_dataset
import utils.networks
import utils.hyperparameters
import utils.optimizers
import itertools

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
if device == 'cuda':
    print(f"CUDA version: {torch.version.cuda}")

parser = argparse.ArgumentParser(description='Classification Experiment')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--model', default='lenet', type=str, help='model: lenet, resnet9, resnet50')
parser.add_argument('--repeats', default=1, type=int, help='Number of repeats for averaging.')
parser.add_argument('--subsample', action='store_true', help='Use less datapoints for train and test.')
parser.add_argument('--pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--verbose', action='store_true',help='verbose flag for all methods')
parser.add_argument('--extra_verbose', action='store_true',help='extra verbose flag for some methods')
parser.add_argument('--progress', action='store_false')
args = parser.parse_args()


# Get dataset
dataset = utils.classification_dataset.load_dataset(name=args.dataset, subsample=args.subsample)

#--- Get hyperparameters from config file
config = utils.hyperparameters.get_config('utils/classification.ini', args.model, args.dataset)
# ---

print('Creating dataloader')
train_dataloader = dataset.trainloader(batch_size=config['bs'])
test_dataloader = dataset.testloader(batch_size=config['bs'])
ood_test_dataloader = dataset.oodtestloader(batch_size=config['bs'])
loss_fn = nn.CrossEntropyLoss()

# Setup directories
res_dir = f"./results/hyperparameters/{args.dataset}_{args.model}/"
ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
if not args.subsample:
    res_dir = res_dir + f"_{time_str}/"
else:
    res_dir = res_dir + f"_s_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if args.verbose:
    print(f'Using {args.dataset} dataset, num train points = {len(dataset.training_data)}')

# Get network loaded
map_net = utils.networks.get_model(args.model, dataset.n_output, dataset.n_channels).to(device)
num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
if args.verbose:
    print(f"Network parameter count: {num_weights}")

# Multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    map_net = nn.DataParallel(map_net)

if args.pre_load:
    print('Using pre-loaded weights!')
    map_net.load_state_dict(torch.load(f'models/{args.model}_trained_{args.dataset}.pt', weights_only=True, map_location=device))
    map_net.eval()
else:
    optimizer, scheduler = utils.optimizers.get_optim_sched(map_net, config['optim'], config['sched'], config['lr'], config['wd'], config['epochs'])
    train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                scheduler=scheduler,epochs=config['epochs'],
                                                                verbose=args.verbose, progress_bar=args.progress)
    torch.save(map_net.state_dict(), f'models/{args.model}_trained_{args.dataset}.pt')

epoch_range = [10]
lr_range = [1e-1]
gamma_range = [0.01,0.03,0.06,0.1]
batchsize_range = [100]
S = 100

dict = {'epoch': [],
        'lr': [],
        'gamma': [],
        'batchsize': [],
        'trainloss': [],
        'trainacc': [],
        'acc': [],
        'ece': [],
        'aucroc': [],
        'varroc-id': [],
        'varroc-ood': []}

for epoch, lr, gamma, batchsize in itertools.product(epoch_range, lr_range, gamma_range, batchsize_range):
    if args.verbose:
        print(f'\nepoch: {epoch}; lr: {lr}; gamma: {gamma}; bs: {batchsize}')

    # Record hyperparameters
    dict['epoch'].append(epoch)
    dict['lr'].append(lr)
    dict['gamma'].append(gamma)
    dict['batchsize'].append(batchsize)

    trainloss_running = 0
    trainacc_running = 0
    acc_running = 0
    ece_running = 0
    aucroc_running = 0
    varrocid_running = 0
    varrocood_running = 0

    for idx in range(args.repeats):
        # Train CUQLS
        cuqls_posterior = Cuqls(map_net, task='classification')
        loss,acc = cuqls_posterior.train(train=dataset.training_data, 
                            batchsize=batchsize, 
                            S=S,
                            scale=gamma, 
                            lr=lr, 
                            epochs=epoch, 
                            mu=0.9,
                            scheduler=False,
                            verbose=args.verbose,
                            extra_verbose=args.extra_verbose,
                            save_weights=f'models/cuqls_{args.model}_{args.dataset}.pt'
                            )
        trainloss_running += loss
        trainacc_running += acc

        # Get predictions
        id_logits = cuqls_posterior.test(dataset.test_data, test_bs=100) 
        id_predictions = id_logits.softmax(dim=2)
        ood_logits = cuqls_posterior.test(dataset.ood_test_data, test_bs=100)
        ood_predictions = ood_logits.softmax(dim=2)

        # Compute metrics
        _, acc, ece, _, auc_roc, _, var_roc_id, var_roc_ood, _ = metrics.compute_metrics(test_dataloader, 
                                                                                        id_predictions.cpu(), 
                                                                                        ood_predictions.cpu(), 
                                                                                        samples=True)
        acc_running += acc
        ece_running += ece
        aucroc_running += auc_roc
        varrocid_running += var_roc_id
        varrocood_running += var_roc_ood



    # Save results
    dict['trainloss'].append(trainloss_running / args.repeat)
    dict['trainacc'].append(trainacc_running / args.repeat)
    dict['acc'].append(acc_running / args.repeats)
    dict['ece'].append(ece_running / args.repeats)
    dict['aucroc'].append(aucroc_running / args.repeats)
    dict['varroc-id'].append(varrocid_running / args.repeats)
    dict['varroc-ood'].append(varrocood_running / args.repeats)

torch.save(dict,res_dir + f'dict.pt')