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
parser.add_argument('--save_var', action='store_true', help='save variances if on (memory consumption)')
parser.add_argument('--pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--progress', action='store_false')
parser.add_argument('--lla_incl', action='store_true', help='if flag is included, lla will run (bugs out with nuqls)')
parser.add_argument('--nuqls_na', action='store_true', help='whether the training data should have data augmentation for NUQLS')
parser.add_argument('--na', action='store_true')
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

    transform_train_noaugmentation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.na:
        training_data = datasets.CIFAR10(
        root="data/CIFAR10",
        train=True,
        download=True,
        transform=transform_train_noaugmentation
    )
    else:
        training_data = datasets.CIFAR10(
        root="data/CIFAR10",
        train=True,
        download=True,
        transform=transform_train
    )

    training_data_na = datasets.CIFAR10(
        root="data/CIFAR10",
        train=True,
        download=True,
        transform=transform_train_noaugmentation
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

n_val = 5000
val_data = torch.utils.data.Subset(training_data,range(n_val))

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

train_dataloader = DataLoader(training_data, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=bs)
ood_test_dataloader = DataLoader(ood_test_data, batch_size=bs)

loss_fn = nn.CrossEntropyLoss()
mse_loss_fn = nn.MSELoss()

# Setup metrics
methods = ['MAP','NUQLS']
train_methods = ['MAP','NUQLS']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'nll': [],
                        'acc': []}
    test_res[m] = {'nll': [],
                  'acc': [],
                  'ece': [],
                  'brier': [],
                  'oodauc': [],
                  'aucroc': [],
                  'time': []}

# Setup directories
res_dir = f"./results/image_classification/{args.dataset}_{args.model}/"

ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
if not args.subsample:
    res_dir = res_dir + f"_{time_str}/"
else:
    res_dir = res_dir + f"_s_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

for ei in tqdm.trange(n_experiment):
    print("\n--- experiment {} ---".format(ei))

    for m in methods:
        t1 = time.time()
        if m == 'MAP':
            if args.model == 'mlp2':
                map_net = model.MLP2()
                map_net.apply(utils.training.init_weights_he)
            elif args.model == 'resnet20':
                map_net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=(args.pre_load and not args.na))
                train_res[m]['nll'].append(0.0103)
                train_res[m]['acc'].append(0.9987)
            elif args.model == 'resnet32':
                map_net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
                train_res[m]['nll'].append(0.00040)
                train_res[m]['acc'].append(0.9995)
            elif args.model == 'resnet44':
                map_net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet44", pretrained=True)
                train_res[m]['nll'].append(0.0024)
                train_res[m]['acc'].append(0.9997)
            elif args.model == 'resnet56':
                map_net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
                train_res[m]['nll'].append(0.0020)
                train_res[m]['acc'].append(0.9998)
            
            map_net.eval()
            num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            print(f"num weights = {num_weights}")
            map_net.to(device)

            if args.model == 'mlp2' or args.model == 'lenet_custom_bn':
                if args.pre_load:
                    print('Using pre-loaded weights!')
                    map_net.load_state_dict(torch.load(f'models/{args.model}_trained_{args.dataset}_na_{args.na}.pt', weights_only=True, map_location=torch.device('cpu')))
                    map_net.eval()
                    train_res[m]['nll'].append(0.0)
                    train_res[m]['acc'].append(1.0)
                else:
                    optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
                    scheduler = None

                    train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                                model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                                scheduler=scheduler,epochs=epochs,
                                                                                verbose=args.verbose, progress_bar=args.progress)
                    train_res[m]['nll'].append(train_loss)
                    train_res[m]['acc'].append(train_acc)
                    torch.save(map_net.state_dict(), f'models/{args.model}_trained_{args.dataset}_na_{args.na}.pt')
            elif args.model == 'resnet20' and not args.pre_load:
                if args.pre_load and args.na:
                    print('Using pre-loaded weights!')
                    map_net.load_state_dict(torch.load(f'models/{args.model}_trained_{args.dataset}_na_{args.na}.pt', weights_only=True, map_location=torch.device('cpu')))
                    map_net.eval()
                    train_res[m]['nll'].append(0.0)
                    train_res[m]['acc'].append(1.0)
                else:
                    optimizer = torch.optim.SGD(map_net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

                    train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                                model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                                scheduler=scheduler,epochs=epochs,
                                                                                verbose=args.verbose, progress_bar=args.progress)
                    train_res[m]['nll'].append(train_loss)
                    train_res[m]['acc'].append(train_acc)
                    torch.save(map_net.state_dict(), f'models/{args.model}_trained_{args.dataset}.pt')

            id_map_logits = pu.test_sampler(map_net, test_data, bs=bs, probit=False)
            ood_map_logits = pu.test_sampler(map_net, ood_test_data, bs=bs, probit=False)

            torch.save(id_map_logits, res_dir + f'map_id_logits.pt')
            torch.save(id_map_logits, res_dir + f'map_ood_logits.pt')

            id_mean = torch.nn.functional.softmax(id_map_logits,dim=1)
            ood_mean = torch.nn.functional.softmax(ood_map_logits,dim=1)

            id_val_map_logits = pu.test_sampler(map_net, val_data, bs=bs, probit=False)
            torch.save(id_val_map_logits, res_dir + f'id_val_map_logits.pt')

        elif m == 'NUQLS':
            nuql = nuqls.classification_parallel_i(map_net)
            if args.nuqls_na and args.model != 'mlp2':
                loss, acc = nuql.train(train=training_data_na, train_bs = nuqls_bs, n_output=n_output, 
                                  S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9, verbose=args.verbose)
            else:
                loss, acc = nuql.train(train=training_data, train_bs = nuqls_bs, n_output=n_output, 
                                  S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9, verbose=args.verbose)
            nuqls_predictions = nuql.test(test=test_data, test_bs=nuqls_bs)
            ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)

            train_res[m]['nll'].append(loss)
            train_res[m]['acc'].append(acc)

            torch.save(nuqls_predictions, res_dir + f'nuqls_id_preds.pt')
            torch.save(ood_nuqls_predictions, res_dir + f'nuqls_ood_preds.pt')

            id_logits = nuqls_predictions
            ood_logits = ood_nuqls_predictions
            id_predictions = nuqls_predictions.softmax(dim=2)
            ood_predictions = ood_nuqls_predictions.softmax(dim=2)
            id_mean = nuqls_predictions.softmax(dim=2).mean(dim=0)
            ood_mean = ood_nuqls_predictions.softmax(dim=2).mean(dim=0)
            id_logit_var = nuqls_predictions.var(0)
            ood_logit_var = ood_nuqls_predictions.var(0)

            nuqls_predictions_val = nuql.test(test=val_data, test_bs=nuqls_bs)
            torch.save(nuqls_predictions_val, res_dir + f'nuqls_id_preds_val.pt')

        t2 = time.time()
        test_res[m]['time'].append(t2-t1)

        p_dist = Categorical(probs=id_mean.to(device))
        test_res[m]['nll'].append(-p_dist.log_prob(test_y.to(device)).mean().item())
        test_res[m]['acc'].append((id_mean.argmax(1).to(device) == test_y.to(device)).type(torch.float).mean().item())
        test_res[m]['ece'].append(metric(id_mean.to(device),test_y.to(device)).cpu().item())
        test_res[m]['brier'].append((mse_loss_fn(id_mean.to(device), torch.nn.functional.one_hot(test_y.to(device),num_classes=n_output))*n_output).item()) # MSELoss takes mean over NxC

        oodauc, aucroc = metrics.auc_metric(id_mean, ood_mean, logits=False)
        test_res[m]['oodauc'].append(oodauc)
        test_res[m]['aucroc'].append(aucroc)

        if m != 'MAP':
            varroc = metrics.aucroc(ood_logit_var.sum(1).detach().cpu(), id_logit_var.sum(1).detach().cpu())
            
        
        ### print (train and) test prediction results
        print(f"\n--- Method {m} ---")
        if m in train_res:
            print("\nTrain Results:")
            print(f"Train Loss: {train_res[m]['nll'][ei]:.3f}; Train Acc: {train_res[m]['acc'][ei]:.1%}")
        print("\nTest Prediction:")
        t = time.strftime("%H:%M:%S", time.gmtime(test_res[m]['time'][ei]))
        print(f"Time h:m:s: {t}")
        print(f"Acc.: {test_res[m]['acc'][ei]:.1%}; ECE: {test_res[m]['ece'][ei]:.1%}; NLL: {test_res[m]['nll'][ei]:.3}; Brier: {test_res[m]['brier'][ei]:.3}")
        print(f"OOD-AUC: {test_res[m]['oodauc'][ei]:.1%}; AUC-ROC: {test_res[m]['aucroc'][ei]:.1%}\n")
        if m != 'MAP':
            # print(f"VARROC: {varroc:.1%}, VARROC_ID: {varroc_id:.1%}\n")
            print(f"VARROC: {varroc:.1%}\n")


## Record results
res_text = res_dir + f"result.txt"
results = open(res_text,'w')
torch.save(train_res,res_dir + f'train_res.pt')
torch.save(test_res,res_dir + f'test_res.pt')

percentage_metrics = ['acc','ece','oodauc','aucroc']

results.write(" --- MAP Training Details --- \n")
results.write(f"epochs: {epochs}; M: {S}; lr: {lr}; weight_decay: {wd}\n")

results.write("\n --- NUQLS Details --- \n")
results.write(f"epochs: {nuqls_epoch}; S: {nuqls_S}; lr: {nuqls_lr}; weight_decay: {nuqls_wd}; init scale: {nuqls_gamma}\n, na: {args.nuqls_na}")

for m in methods:
    if n_experiment > 1:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                if k in percentage_metrics:
                    results.write(f"{k}: {np.mean(train_res[m][k]):.1%} +- {np.std(train_res[m][k]):.1%} \n")
                else:
                    results.write(f"{k}: {np.mean(train_res[m][k]):.3f} +- {np.std(train_res[m][k]):.3f} \n")
        results.write("\n - Test Prediction: - \n")
        for k in test_res[m].keys():
            if k == 'time':
                t = time.strftime("%H:%M:%S", time.gmtime(np.mean(test_res[m][k])))
                results.write(f"{k}: {t}\n")
            elif k in percentage_metrics:
                results.write(f"{k}: {np.mean(test_res[m][k]):.1%} +- {np.std(test_res[m][k]):.1%} \n")
            else:
                results.write(f"{k}: {np.mean(test_res[m][k]):.3f} +- {np.std(test_res[m][k]):.3f} \n")
    else:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                if k in percentage_metrics:
                    results.write(f"{k}: {train_res[m][k][0]:.1%}\n")
                else:
                    results.write(f"{k}: {train_res[m][k][0]:.3f}\n")
        results.write("\n - Test Prediction: - \n")
        for k in test_res[m].keys():
            if k == 'time':
                t = time.strftime("%H:%M:%S", time.gmtime(test_res[m][k][0]))
                results.write(f"{k}: {t}\n")
            elif k in percentage_metrics:
                results.write(f"{k}: {test_res[m][k][0]:.1%}\n")
            else:
                results.write(f"{k}: {test_res[m][k][0]:.3f}\n")

results.close()