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
import posteriors.nuqlsPosterior.nuqls as nqls

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

parser = argparse.ArgumentParser(description='Classification Experiment')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--model', default='lenet', type=str, help='model: lenet, resnet9, resnet50')
parser.add_argument('--subsample', action='store_true', help='Use less datapoints for train and test.')
parser.add_argument('--verbose', action='store_true',help='verbose flag for all methods')
parser.add_argument('--save_var', action='store_true', help='save variances if on (memory consumption)')
parser.add_argument('--pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--progress', action='store_false')
parser.add_argument('--lla_incl', action='store_true', help='if flag is included, lla will run (bugs out with nuqls)')
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

elif args.dataset=='svhn':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    ])

    training_data = datasets.SVHN(
        root="data/SVHN",
        split='train',
        download=True,
        transform=transform_train
    )

    test_data = datasets.SVHN(
        root="data/SVHN",
        split='test',
        download=True,
        transform=transform_test
    )

    transform_test_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ood_test_data = datasets.CIFAR10(
        root="data/CIFAR10",
        train=False,
        download=True,
        transform=transform_test_cifar
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
    N_TRAIN = 5000
    N_TEST = 100
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

# Setup metrics
if args.lla_incl:
    methods = ['MAP','LLA']
    train_methods = ['MAP']
else:
    # methods = ['MAP','NUQLS','DE','SWAG','MC']
    methods = ['MAP','NUQLS_TEST']
    # methods = ['MAP','SWAG','MC']
    train_methods = ['MAP','NUQLS_TEST','NUQLS','DE','MC']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'nll': [],
                        'acc': []}
    test_res[m] = {'nll': [],
                  'acc': [],
                  'ece': [],
                  'oodauc': [],
                  'aucroc': [],
                  'varroc': [],
                  'time': []}

prob_var_dict = {}
prediction_dict = {}

# Setup directories
res_dir = f"./results/image_classification/{args.dataset}_{args.model}/"
# if not args.subsample:
#     method_dir = f"./method/{args.dataset}/{args.model}"
# else:
#     method_dir = f"./method/{args.dataset}_{N_TRAIN}_{N_TEST}/{args.model}/"
#     dataset_dir = {'train': training_data,
#                    'test': test_data,
#                    'ood_test': ood_test_data,
#                    'val': val_data,}

ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
if not args.subsample:
    res_dir = res_dir + f"_{time_str}/"
else:
    res_dir = res_dir + f"_s_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

print(f'Using {args.dataset} dataset, num train points = {len(training_data)}')

for ei in tqdm.trange(n_experiment):
    print("\n--- experiment {} ---".format(ei))

    for m in methods:
        t1 = time.time()
        if m == 'MAP':
            if args.model == 'lenet':
                map_net = model.LeNet5()
            if args.model == 'lenet_custom':
                map_net = model.LeNet5_custom()
            elif args.model == 'resnet9':
                map_net = model.ResNet9(in_channels=n_channels, num_classes=n_output)
            elif args.model == 'wrn':
                map_net = model.WRN(depth=28, widening_factor=5, num_classes = n_output)
            elif args.model == 'resnet50':
                map_net = model.ResNet50(in_channels=n_channels, num_classes = n_output)

            if args.model == 'lenet_custom':
                map_net.apply(utils.training.init_weights_he)
            else:
                map_net.apply(utils.training.init_weights)
            map_net.eval()
            num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            print(f"num weights = {num_weights}")

            # Multiple GPUs
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                map_net = nn.DataParallel(map_net)
            
            map_net.to(device)

            if args.pre_load:
                print('Using pre-loaded weights!')
                map_net.load_state_dict(torch.load(f'models/{args.model}_trained_{args.dataset}.pt', weights_only=True))
                map_net.eval()
                map_net.to(device)
                train_res[m]['nll'].append(0.0)
                train_res[m]['acc'].append(1.0)
            else:
                if args.dataset == 'mnist' or args.dataset == 'fmnist':
                    if args.model == 'lenet_custom':
                        optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
                        scheduler = None
                    else:
                        optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
                elif args.dataset == 'cifar10':
                    if args.model == 'resnet9':
                        optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=wd)
                        scheduler = None
                    elif args.model == 'resnet50':
                        optimizer = torch.optim.SGD(map_net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
                elif args.dataset == 'svhn':
                    if args.model == 'resnet50':
                        optimizer = torch.optim.SGD(map_net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
                elif args.dataset == 'cifar100':
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

            id_mean = torch.nn.functional.softmax(id_map_logits,dim=1)
            ood_mean = torch.nn.functional.softmax(ood_map_logits,dim=1)

        elif m == 'NUQLS_TEST':
            nuqls_posterior = nqls.Nuqls(map_net, task='classification')
            loss,acc = nuqls_posterior.train(train=training_data, 
                                train_bs=nuqls_bs, 
                                n_output=n_output,
                                S=nuqls_S,
                                scale=nuqls_gamma, 
                                lr=nuqls_lr, 
                                epochs=nuqls_epoch, 
                                mu=0.9,
                                verbose=args.verbose)
            train_res[m]['nll'].append(loss)
            train_res[m]['acc'].append(acc)

            id_logits = nuqls_posterior.test(test_data, test_bs=nuqls_bs) 
            id_predictions = id_logits.softmax(dim=2)
            ood_logits = nuqls_posterior.test(ood_test_data, test_bs=nuqls_bs)
            ood_predictions = ood_logits.softmax(dim=2)

            id_mean = id_predictions.mean(dim=0); id_logit_var = id_logits.var(0)
            ood_mean = ood_predictions.mean(dim=0); ood_logit_var = ood_logits.var(0)
            
            
        elif m == 'NUQLS':
            nuql = nuqls.classification_parallel_i(map_net)
            loss, acc = nuql.train(train=training_data, train_bs = nuqls_bs, n_output=n_output, S = nuqls_S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9, verbose=args.verbose)
            nuqls_predictions = nuql.test(test=test_data, test_bs=nuqls_bs)
            ood_nuqls_predictions = nuql.test(test=ood_test_data, test_bs=nuqls_bs)

            train_res[m]['nll'].append(loss)
            train_res[m]['acc'].append(acc)

            id_predictions = nuqls_predictions.softmax(dim=2)
            ood_predictions = ood_nuqls_predictions.softmax(dim=2)
            id_mean = nuqls_predictions.softmax(dim=2).mean(dim=0)
            ood_mean = ood_nuqls_predictions.softmax(dim=2).mean(dim=0)
            id_logit_var = nuqls_predictions.var(0)
            ood_logit_var = ood_nuqls_predictions.var(0)


        elif m == 'DE':
            model_list = []
            opt_list = []
            sched_list = []
            for i in range(S):
                if args.model == 'lenet':
                    model_list.append(model.LeNet5().to(device))
                elif args.model == 'resnet9':
                    model_list.append(model.ResNet9(in_channels=n_channels, num_classes = n_output).to(device))
                elif args.model == 'wrn':
                    model_list.append(model.WRN(depth=28, widening_factor=5, num_classes = n_output).to(device))
                elif args.model == 'resnet50':
                    model_list.append(model.ResNet50(in_channels=n_channels, num_classes = n_output).to(device))
                model_list[i].apply(utils.training.init_weights)
                if args.model == 'resnet50':
                    opt_list.append(torch.optim.SGD(model_list[i].parameters(), lr = lr, momentum=0.9, weight_decay = wd))
                else:
                    opt_list.append(torch.optim.Adam(model_list[i].parameters(), lr = lr, weight_decay = wd))
                if args.model == 'resnet9' and args.dataset == 'cifar10':
                    sched_list.append(None)
                else:
                    sched_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt_list[i], T_max = epochs))

            de_train_loss = 0
            de_train_acc = 0

            if args.progress:
                pbar = tqdm.trange(S)
            else:
                pbar = range(S)

            for i in pbar:
                train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader, test_loader=test_dataloader,
                                                                            model=model_list[i],loss_fn=loss_fn,optimizer=opt_list[i],
                                                                            scheduler=sched_list[i],epochs=epochs,verbose=args.verbose, progress_bar=False)
                de_train_loss += train_loss
                de_train_acc += train_acc
            de_train_loss /= S
            de_train_acc /= S

            train_res[m]['nll'].append(de_train_loss)
            train_res[m]['acc'].append(de_train_acc)
            
            ### Deep ensembles inference

            id_mean, _, id_predictions = pu.ensemble_sampler(dataset=test_data,M=S,      # id_predictions -> S x N x C
                                                    models=model_list,n_output=n_output,
                                                    bs=bs)
            ood_mean, _, ood_predictions = pu.ensemble_sampler(dataset=ood_test_data,M=S,    # ood_predictions -> : S x N x C
                                                    models=model_list,n_output=n_output,
                                                    bs=bs)
    
        elif m == 'eNUQLS':
            S = nuqls_S
            id_preds = []
            ood_preds = []
            for i in range(S):
                if S > 10:
                    nuqls_method = nuqls.classification_parallel(net = model_list[i], train = training_data, S = S, epochs=nuqls_epoch, lr=nuqls_lr, n_output=n_output, 
                                                            bs=nuqls_bs, bs_test=nuqls_bs, init_scale=nuqls_gamma)
                    nuqls_predictions, ood_nuqls_predictions, res = nuqls_method.method(test_data, ood_test = ood_test_data, mu=0.9, 
                                                                                        weight_decay=nuqls_wd, verbose=args.verbose, 
                                                                                        progress_bar=args.progress, gradnorm=True) # S x N x C
                    del nuqls_method
                else:
                    nuqls_predictions, ood_nuqls_predictions, res = nuqls.series_method(net = model_list[i], train_data = training_data, test_data = test_data, 
                                                                                ood_test_data=ood_test_data, train_bs = nuqls_bs, test_bs = nuqls_bs, 
                                                                                S = S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9, 
                                                                                wd = nuqls_wd, verbose = False, progress_bar = True) # S x N x C
                print(res['loss'])
                print(res['acc'])

                train_res[m]['nll'].append(res['loss'].detach().cpu().item())
                train_res[m]['acc'].append(res['acc'])

                id_predictions = nuqls_predictions.softmax(dim=2)
                id_preds.append(id_predictions)
                ood_predictions = ood_nuqls_predictions.softmax(dim=2)
                ood_preds.append(ood_predictions)
            id_predictions = torch.cat(id_preds,dim=0)
            ood_predictions = torch.cat(ood_preds,dim=0)

            id_mean = id_predictions.mean(dim=0)
            ood_mean = ood_predictions.mean(dim=0)

        elif m == 'LLA':
            ## LLA definitions
            def predict(dataloader, la, link='probit'):
                py = []
                for x, _ in dataloader:
                    py.append(la(x.to(device), pred_type="glm", link_approx=link))
                return torch.cat(py).cpu()

            la = Laplace(map_net, "classification",
                        subset_of_weights="last_layer",
                        hessian_structure="kron")
            laplace_train_loader = DataLoader(training_data,batch_size=bs)
            laplace_test_loader = DataLoader(test_data,batch_size=bs)
            laplace_ood_test_loader = DataLoader(ood_test_data,batch_size=bs)
            laplace_val_loader = DataLoader(val_data,batch_size=bs)
            la.fit(laplace_train_loader)
            # la.optimize_prior_precision(
            #     method="gridsearch",
            #     pred_type='glm',
            #     val_loader = laplace_val_loader,
            #     log_prior_prec_min=-2,
            #     log_prior_prec_max = 2,
            #     grid_size=20,
            #     link_approx='probit',
            #     progress_bar=args.progress
            # )
            la.optimize_prior_precision(
                method="marglik",
                pred_type='glm',
                link_approx='probit',
                progress_bar=args.progress
            )


            T = 1000
            id_mean, _, id_predictions = pu.lla_sampler(dataset=test_data, 
                                                              model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                              bs = bs)  # id_predictions -> S x N x C

            ood_mean, _, ood_predictions = pu.lla_sampler(dataset=ood_test_data, 
                                                                    model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                                    bs = bs)  # ood_predictions -> S x N x C
            
        elif m == 'LLA_S':
            zeta, scale = lla_s.classification(net=map_net, train=training_data, train_bs=50, k=9, n_output=n_output, 
                                alpha0=1, alpha_it=5, post_lr=1e-4, post_epochs=100, lin_lr=1e-2, lin_epochs=100, compute_exact=False)
            
            preds = lla_s.classification_prediction(net=map_net, zeta=zeta, test=test_data, test_bs=50, scale=scale) # S x N x C
            ood_preds = lla_s.classification_prediction(net=map_net, zeta=zeta, test=ood_test_data, test_bs=50, scale=scale) # S x N x C

            id_mean = preds.softmax(dim=2).mean(dim=0)
            ood_mean = ood_preds.softmax(dim=2).mean(dim=0)
            id_predictions = preds.softmax(dim=2)
            ood_predictions = ood_preds.softmax(dim=2)

        elif m == 'VaLLA':
            num_inducing = 100
            seed = 0
            prior_std = 1
            bb_alpha = 1
            iterations = 5000
            val_loader = DataLoader(val_data, batch_size=50)
            fixed_prior = False
            dtype = torch.float64
            generator = torch.Generator(device=device)
            generator.manual_seed(2147483647)
            lla_probit = True

            Z = []
            classes = []
            for c in range(n_output):
                # s = training_data.inputs[training_data.targets.flatten() == c]
                s = train_x[train_y == c]
                z = kmeans2(s.reshape(s.shape[0], -1), 
                            num_inducing//n_output, minit="points", 
                            seed=seed)[0]
                z = z.reshape(num_inducing//n_output, 
                            *train_x.shape[1:])

                Z.append(z)
                classes.append(np.ones(num_inducing//n_output) * c)
            Z = np.concatenate(Z)
            classes = np.concatenate(classes)

            from posteriors.valla.src.backpack_interface import BackPackInterface

            valla = VaLLAMultiClassBackend(
                map_net,
                Z,
                backend = BackPackInterface(map_net, n_output),
                prior_std=prior_std,
                num_data=len(training_data),
                output_dim=n_output,
                track_inducing_locations=True,
                inducing_classes=classes,
                y_mean=0.0,
                y_std=1.0,
                alpha = bb_alpha,
                device=device,
                dtype=dtype,
                #seed = args.seed
            )

            opt = torch.optim.Adam(valla.parameters(recurse=False), lr=1e-3)
            valla_train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)

            loss, val_loss = fit(
                valla,
                valla_train_dataloader,
                opt,
                val_metrics=SoftmaxClassification,
                val_steps=valla.num_data//bs,
                val_generator = val_loader,
                use_tqdm=args.verbose,
                return_loss=True,
                iterations=iterations,
                device=device,
                dtype = dtype
            )

            # Get MC-sample predictions to get softmax variance
            id_preds = []
            for x,y in test_dataloader:
                _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
                
                chol = psd_safe_cholesky(Fvar)
                z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = generator, device = device,
                                    dtype = dtype)
                preds = Fmean + torch.einsum("sna, nab -> snb", z, chol) # S n X n C
                id_preds.append(preds.detach())
            id_preds = torch.cat(id_preds,dim=1)

            ood_preds = []
            for x,y in ood_test_dataloader:
                _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
                chol = psd_safe_cholesky(Fvar)
                z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = generator, device = device,
                                    dtype = dtype)
                preds = Fmean + torch.einsum("sna, nab -> snb", z, chol) # S n X n C
                ood_preds.append(preds.detach())
            ood_preds = torch.cat(ood_preds,dim=1)

            id_mean = id_preds.softmax(dim=2).mean(dim=0)
            ood_mean = ood_preds.softmax(dim=2).mean(dim=0)
            id_predictions = id_preds.softmax(dim=2)
            ood_predictions = ood_preds.softmax(dim=2)

            # Using probit approximation may lead to better results
            if lla_probit:
                id_preds = []
                for x,y in test_dataloader:
                    _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
                    scaled_logits = Fmean/torch.sqrt( 1 + torch.pi/8 * torch.diagonal(Fvar, dim1 = 1, dim2 = 2))
                    id_preds.append(scaled_logits.detach())
                id_preds = torch.cat(id_preds,dim=0)

                ood_preds = []
                for x,y in ood_test_dataloader:
                    _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
                    scaled_logits = Fmean/torch.sqrt( 1 + torch.pi/8 * torch.diagonal(Fvar, dim1 = 1, dim2 = 2))
                    ood_preds.append(scaled_logits.detach())
                ood_preds = torch.cat(ood_preds,dim=0)

                id_mean = id_preds.softmax(dim=1)
                ood_mean = ood_preds.softmax(dim=1)

        elif m == 'SWAG':
            if args.dataset == 'cifar100':
                swag_lr = lr
                swag_wd = 1e-4
            else:
                swag_lr = lr*1e2
                swag_wd = 0
            swag_net = swag.SWAG(map_net,epochs = epochs,lr = swag_lr, cov_mat = True,
                                max_num_models=S, wd=swag_wd)
            swag_net.train_swag(train_dataloader=train_dataloader,progress_bar=args.progress)

            T = 100
            id_mean, _, id_predictions = pu.swag_sampler(dataset=test_data,model=swag_net,T=T,n_output=n_output,bs=1000) # id_predictions -> S x N x C
            ood_mean, _, ood_predictions = pu.swag_sampler(dataset=ood_test_data,model=swag_net,T=T,n_output=n_output,bs=1000) # ood_predictions -> S x N x C
        
        elif m == 'MC':
            p = 0.1

            if args.model == 'lenet':
                mc_net = model.LeNet5_Dropout(p=p).to(device)
            elif args.model == 'resnet9':
                mc_net = model.ResNet9(in_channels=n_channels, num_classes=n_output, p=p).to(device)
            elif args.model == 'wrn':
                mc_net = model.WRN(depth=28, widening_factor=5, num_classes = n_output, drop_rate=p).to(device)
            elif args.model == 'resnet50':
                mc_net = model.ResNet50(in_channels=n_channels, num_classes = n_output, p = p).to(device)
            mc_net.apply(utils.training.init_weights)
            mc_net.eval()
            num_weights = sum(p.numel() for p in mc_net.parameters() if p.requires_grad)

            if args.model == 'resnet50':
                optimizer = torch.optim.SGD(mc_net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            else:
                optimizer = torch.optim.Adam(mc_net.parameters(), lr=lr, weight_decay=wd)
            if args.model == 'resnet9' and args.dataset == 'cifar10':
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

            train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                        model=mc_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                        scheduler=scheduler,epochs=epochs,
                                                                        verbose=False, train_mode=False, progress_bar=args.progress)
            train_res[m]['nll'].append(train_loss)
            train_res[m]['acc'].append(train_acc)
            
            T = 100
            id_mean, _, id_predictions = pu.dropout_sampler(dataset=test_data,model=mc_net,T=T,n_output=n_output,bs=bs) # id_predictions -> S x N x C
            ood_mean, _, ood_predictions = pu.dropout_sampler(dataset=ood_test_data,model=mc_net,T=T,n_output=n_output,bs=bs) # ood_predictions -> S x N x C

        t2 = time.time()
        test_res[m]['time'].append(t2-t1)

        p_dist = Categorical(probs=id_mean.to(device))
        test_res[m]['nll'].append(-p_dist.log_prob(test_y.to(device)).mean().item())
        test_res[m]['acc'].append((id_mean.argmax(1).to(device) == test_y.to(device)).type(torch.float).mean().item())
        test_res[m]['ece'].append(metric(id_mean.to(device),test_y.to(device)).cpu().item())

        oodauc, aucroc = metrics.auc_metric(id_mean, ood_mean, logits=False)
        test_res[m]['oodauc'].append(oodauc)
        test_res[m]['aucroc'].append(aucroc)

        if m != 'MAP':
            varroc = metrics.aucroc(ood_logit_var.sum(1).detach().cpu(), id_logit_var.sum(1).detach().cpu())
            test_res[m]['varroc'].append(varroc)

        ### print (train and) test prediction results
        print(f"\n--- Method {m} ---")
        if m in train_res:
            print("\nTrain Results:")
            print(f"Train Loss: {train_res[m]['nll'][ei]:.3f}; Train Acc: {train_res[m]['acc'][ei]:.1%}")
        print("\nTest Prediction:")
        t = time.strftime("%H:%M:%S", time.gmtime(test_res[m]['time'][ei]))
        print(f"Time h:m:s: {t}")
        print(f"Acc.: {test_res[m]['acc'][ei]:.1%}; ECE: {test_res[m]['ece'][ei]:.1%}; NLL: {test_res[m]['nll'][ei]:.3}")
        if m == 'MAP':
            print(f"OOD-AUC: {test_res[m]['oodauc'][ei]:.1%}; AUC-ROC: {test_res[m]['aucroc'][ei]:.1%}\n")
        else:
            print(f"OOD-AUC: {test_res[m]['oodauc'][ei]:.1%}; AUC-ROC: {test_res[m]['aucroc'][ei]:.1%}; VARROC: {test_res[m]['varroc'][ei]:.1%}\n") 

        if m != 'MAP':
            if args.save_var:
                prob_var_dict[m] = metrics.sort_probabilies(id_predictions.to('cpu'), ood_predictions.to('cpu'), test_data=test_data)

    # Save predictions, variances for plotting
    if args.save_var:
        prob_var_dict = metrics.add_baseline(prob_var_dict,test_data,ood_test_data)
        torch.save(prob_var_dict,res_dir + f"prob_var_dict_{ei}.pt")

        metrics.plot_vmsp(prob_dict=prob_var_dict,
                          title=f'{args.dataset} {args.model}',
                          save_fig=res_dir + f"vmsp_plot.pdf")

## Record results
res_text = res_dir + f"result.txt"
results = open(res_text,'w')
torch.save(train_res,res_dir + f'train_res.pt')
torch.save(test_res,res_dir + f'test_res.pt')

percentage_metrics = ['acc','ece','oodauc','aucroc','varroc']

results.write(" --- MAP Training Details --- \n")
results.write(f"epochs: {epochs}; M: {S}; lr: {lr}; weight_decay: {wd}\n")

results.write("\n --- NUQLS Details --- \n")
results.write(f"epochs: {nuqls_epoch}; S: {nuqls_S}; lr: {nuqls_lr}; weight_decay: {nuqls_wd}; init scale: {nuqls_gamma}\n")

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
            elif k == 'varroc' and m == 'MAP':
                continue
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
            elif k == 'varroc' and m == 'MAP':
                continue
            elif k in percentage_metrics:
                results.write(f"{k}: {test_res[m][k][0]:.1%}\n")
            else:
                results.write(f"{k}: {test_res[m][k][0]:.3f}\n")

results.close()