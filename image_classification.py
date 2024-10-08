import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from tqdm import tqdm
import definitions_2 as df2
import posteriors.nuqls as nuqls
from tqdm import tqdm
import numpy as np
import models as model
from torch import nn
from torch.utils.data import DataLoader
from laplace import Laplace
import os
import posteriors.swag as swag
from posteriors.lla.likelihoods import Categorical
import time
import datetime
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAUROC
import argparse
import warnings

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
parser.add_argument('--n_experiment',default=5,type=int,help='number of experiments')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--model', default='lenet', type=str, help='model: lenet, resnet9, resnet50')
parser.add_argument('--subsample', action='store_true', help='Use less datapoints for train and test.')
parser.add_argument('--verbose', action='store_true',help='verbose flag for all methods')
parser.add_argument('--save_var', action='store_true', help='save variances if on (memory consumption)')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs for training networks.')
parser.add_argument('--wd',default=1e-4,type=float,help='wd for training networks.')
parser.add_argument('--lr',default=5e-3,type=float,help='lr for training networks.')
parser.add_argument('--bs', default=100, type=int, help='batch size for training networks.')
parser.add_argument('--nuqls_epoch',default=1000,type=int,help='epochs for nuqls.')
parser.add_argument('--nuqls_S',default=10,type=int,help='num realisations for nuqls.')
parser.add_argument('--nuqls_lr',default=1e-1,type=float,help='lr for nuqls.')
parser.add_argument('--nuqls_bs',default=100,type=int,help='batch size for nuqls.')
parser.add_argument('--nuqls_wd',default=0,type=float,help='wd for nuqls.')
parser.add_argument('--nuqls_gamma',default=1,type=float,help='init scale for nuqls.')
parser.add_argument('--progress', action='store_false')
parser.add_argument('--lla_incl', action='store_true', help='if flag is included, lla will run (bugs out with nuqls)')
args = parser.parse_args()


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

elif args.dataset=='cifar':
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

    # train_data, val_data = torch.utils.data.random_split(training_data,[50000,10000])
    val_data = test_data
    n_output = 10
    n_channels = 3

if args.subsample:
    N_TRAIN = 1000
    N_TEST = 100
    training_data = torch.utils.data.Subset(training_data,range(N_TRAIN))
    test_data = torch.utils.data.Subset(test_data,range(N_TEST))
    val_data = torch.utils.data.Subset(val_data,range(N_TEST))
    ood_test_data = torch.utils.data.Subset(ood_test_data,range(N_TEST))


metric = MulticlassCalibrationError(num_classes=n_output,n_bins=10,norm='l1')
full_test_dataloader = DataLoader(test_data, len(test_data))
test_x,test_y = next(iter(full_test_dataloader))

full_ood_dataloader = DataLoader(ood_test_data, len(ood_test_data))
ood_test_x,ood_test_y = next(iter(full_ood_dataloader))

learning_rate = args.lr
batch_size = args.bs
epochs = args.epochs
weight_decay = args.wd

M = 10
lla_batch_size = 50

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)
ood_test_dataloader = DataLoader(ood_test_data, batch_size)

loss_fn = nn.CrossEntropyLoss()

# Setup metrics
if args.lla_incl:
    methods = ['MAP','LLA-LL-KFAC']
    train_methods = ['MAP']
else:
    methods = ['MAP','NUQLs','DE','eNUQLs','SWAG','MC-Dropout']
    train_methods = ['MAP','NUQLs','DE','eNUQLs','MC-Dropout']
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
                  'time': []}

prob_var_dict = {}
prediction_dict = {}

# Setup directories
res_dir = f"./results/image_classification/{args.dataset}_{args.model}"

ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
if not args.subsample:
    res_dir = res_dir + f"_{time_str}/"
else:
    res_dir = res_dir + f"_s_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

for ei in tqdm(range(args.n_experiment)):
    print("\n--- experiment {} ---".format(ei))

    for m in methods:
        t1 = time.time()
        if m == 'MAP':
            if args.model == 'lenet':
                map_net = model.LeNet5().to(device)
            elif args.model == 'resnet9':
                map_net = model.ResNet9(in_channels=n_channels, num_classes=n_output).to(device)
            elif args.model == 'wrn':
                map_net = model.WRN(depth=28, widening_factor=5, num_classes = n_output).to(device)
            elif args.model == 'resnet50':
                map_net = model.ResNet50(in_channels=n_channels, num_classes = n_output).to(device)
            map_net.apply(df2.init_weights)
            map_net.eval()
            num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            print(f"num weights = {num_weights}")
            
            if args.model == 'resnet9' and args.dataset == 'cifar':
                optimizer = torch.optim.Adam(map_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = None
            elif args.model == 'resnet50':
                # optimizer = torch.optim.Adam(map_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
                optimizer = torch.optim.SGD(map_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
            else:
                optimizer = torch.optim.Adam(map_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

            train_loss, train_acc, _, _ = df2.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                        model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                        scheduler=scheduler,epochs=epochs,
                                                                        verbose=args.verbose, progress_bar=args.progress)
            train_res[m]['nll'].append(train_loss)
            train_res[m]['acc'].append(train_acc)

            id_mean = df2.test_sampler(map_net, test_data, bs=batch_size, probit=True)
            ood_mean = df2.test_sampler(map_net, ood_test_data, bs=batch_size, probit=True)

        elif m == 'NUQLs':
            S = args.nuqls_S
            if S > 10:
                nuqls_linear_method = nuqls.linear_nuqls_c(net = map_net, train = training_data, S = S, epochs=args.nuqls_epoch, lr=args.nuqls_lr, n_output=n_output, 
                                                        bs=args.nuqls_bs, bs_test=args.nuqls_bs, init_scale=args.nuqls_gamma)
                nuqls_predictions, ood_nuqls_predictions, res = nuqls_linear_method.method(test_data, ood_test = ood_test_data, mu=0.9, 
                                                                                    weight_decay=args.nuqls_wd, verbose=args.verbose, progress_bar=args.progress, gradnorm=True, 
                                                                                    ) # S x N x C
                del nuqls_linear_method
            else:
                nuqls_predictions, ood_nuqls_predictions, res = nuqls.linear_sampling(net = map_net, train_data = training_data, test_data = test_data, 
                                                                            ood_test_data=ood_test_data, train_bs = args.nuqls_bs, test_bs = args.nuqls_bs, 
                                                                            S = S, scale=args.nuqls_gamma, lr=args.nuqls_lr, epochs=args.nuqls_epoch, mu=0.9, 
                                                                            wd = args.nuqls_wd, verbose = False, progress_bar = True) # S x N x C
            print(res['loss'])
            print(res['acc'])

            train_res[m]['nll'].append(res['loss'])
            train_res[m]['acc'].append(res['acc'])

            id_mean = nuqls_predictions.softmax(dim=2).mean(dim=0)
            id_var = nuqls_predictions.softmax(dim=2).var(dim=0)
            ood_mean = ood_nuqls_predictions.softmax(dim=2).mean(dim=0)
            ood_var = ood_nuqls_predictions.softmax(dim=2).var(dim=0)
            id_predictions = nuqls_predictions.softmax(dim=2)
            ood_predictions = ood_nuqls_predictions.softmax(dim=2)

        elif m == 'DE':
            model_list = []
            opt_list = []
            sched_list = []
            for i in range(M):
                if args.model == 'lenet':
                    model_list.append(model.LeNet5().to(device))
                elif args.model == 'resnet9':
                    model_list.append(model.ResNet9(in_channels=n_channels, num_classes = n_output).to(device))
                elif args.model == 'wrn':
                    model_list.append(model.WRN(depth=28, widening_factor=5, num_classes = n_output).to(device))
                elif args.model == 'resnet50':
                    model_list.append(model.ResNet50(in_channels=n_channels, num_classes = n_output).to(device))
                model_list[i].apply(df2.init_weights)
                if args.model == 'resnet50':
                    opt_list.append(torch.optim.SGD(model_list[i].parameters(), lr = learning_rate, momentum=0.9, weight_decay = weight_decay))
                else:
                    opt_list.append(torch.optim.Adam(model_list[i].parameters(), lr = learning_rate, weight_decay = weight_decay))
                if args.model == 'resnet9' and args.dataset == 'cifar':
                    sched_list.append(None)
                else:
                    sched_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt_list[i], T_max = epochs))

            de_train_loss = 0
            de_train_acc = 0
            for i in range(M):
                train_loss, train_acc, _, _ = df2.training(train_loader=train_dataloader, test_loader=test_dataloader,
                                                                            model=model_list[i],loss_fn=loss_fn,optimizer=opt_list[i],
                                                                            scheduler=sched_list[i],epochs=epochs,verbose=args.verbose, progress_bar=args.progress)
                de_train_loss += train_loss
                de_train_acc += train_acc
            de_train_loss /= M
            de_train_acc /= M

            train_res[m]['nll'].append(de_train_loss)
            train_res[m]['acc'].append(de_train_acc)
            
            ### Deep ensembles inference

            id_mean, id_var, id_predictions = df2.ensemble_sampler(dataset=test_data,M=M,      # id_predictions -> S x N x C
                                                    models=model_list,n_output=n_output,
                                                    bs=batch_size)
            ood_mean, ood_var, ood_predictions = df2.ensemble_sampler(dataset=ood_test_data,M=M,    # ood_predictions -> : S x N x C
                                                    models=model_list,n_output=n_output,
                                                    bs=batch_size)
    
        elif m == 'eNUQLs':
            S = args.nuqls_S
            id_preds = []
            ood_preds = []
            for i in range(M):
                if S > 10:
                    nuqls_linear_method = nuqls.linear_nuqls_c(net = model_list[i], train = training_data, S = S, epochs=args.nuqls_epoch, lr=args.nuqls_lr, n_output=n_output, 
                                                            bs=args.nuqls_bs, bs_test=args.nuqls_bs, init_scale=args.nuqls_scale)
                    nuqls_predictions, ood_nuqls_predictions, res = nuqls_linear_method.method(test_data, ood_test = ood_test_data, mu=0.9, 
                                                                                        weight_decay=args.nuqls_wd, verbose=args.verbose, 
                                                                                        progress_bar=args.progress, gradnorm=True) # S x N x C
                    del nuqls_linear_method
                else:
                    nuqls_predictions, ood_nuqls_predictions, res = nuqls.linear_sampling(net = model_list[i], train_data = training_data, test_data = test_data, 
                                                                                ood_test_data=ood_test_data, train_bs = args.nuqls_bs, test_bs = args.nuqls_bs, 
                                                                                S = S, scale=args.nuqls_scale, lr=args.nuqls_lr, epochs=args.nuqls_epoch, mu=0.9, 
                                                                                wd = args.nuqls_wd, verbose = False, progress_bar = True) # S x N x C
                print(res['loss'])
                print(res['acc'])

                train_res[m]['nll'].append(res['loss'])
                train_res[m]['acc'].append(res['acc'])

                id_predictions = nuqls_predictions.softmax(dim=2)
                id_preds.append(id_predictions)
                ood_predictions = ood_nuqls_predictions.softmax(dim=2)
                ood_preds.append(ood_predictions)
            id_predictions = torch.cat(id_preds,dim=0)
            ood_predictions = torch.cat(ood_preds,dim=0)

            id_mean = id_predictions.mean(dim=0)
            id_var = id_predictions.var(dim=0)
            ood_mean = ood_predictions.mean(dim=0)
            ood_var = ood_predictions.var(dim=0)

        elif m == 'LLA-LL-KFAC':
            ## LLA definitions
            def predict(dataloader, la, link='probit'):
                py = []
                for x, _ in dataloader:
                    py.append(la(x.to(device), pred_type="glm", link_approx=link))
                return torch.cat(py).cpu()

            la = Laplace(map_net, "classification",
                        subset_of_weights="last_layer",
                        hessian_structure="kron")
            laplace_train_loader = DataLoader(training_data,batch_size=lla_batch_size)
            laplace_test_loader = DataLoader(test_data,batch_size=lla_batch_size)
            laplace_ood_test_loader = DataLoader(ood_test_data,batch_size=lla_batch_size)
            laplace_val_loader = DataLoader(val_data,batch_size=lla_batch_size)
            la.fit(laplace_train_loader)
            la.optimize_prior_precision(
                method="gridsearch",
                pred_type='glm',
                val_loader = laplace_val_loader,
                log_prior_prec_min=-2,
                log_prior_prec_max = 2,
                grid_size=20,
                link_approx='probit',
                progress_bar=args.progress
            )

            T = 1000
            id_mean, id_var, id_predictions = df2.lla_sampler(dataset=test_data, 
                                                              model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                              bs = batch_size)  # id_predictions -> S x N x C

            ood_mean, ood_var, ood_predictions = df2.lla_sampler(dataset=ood_test_data, 
                                                                    model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                                    bs = batch_size)  # ood_predictions -> S x N x C

        elif m == 'SWAG':
            swag_net = swag.SWAG(map_net,epochs = epochs,lr = learning_rate*1e2, cov_mat = True,
                                max_num_models=M)
            swag_net.train_swag(train_dataloader,args.progress)

            T = 100
            id_mean, id_var, id_predictions = df2.swag_sampler(dataset=test_data,model=swag_net,T=T,n_output=n_output,bs=batch_size) # id_predictions -> S x N x C
            ood_mean, ood_var, ood_predictions = df2.swag_sampler(dataset=ood_test_data,model=swag_net,T=T,n_output=n_output,bs=batch_size) # ood_predictions -> S x N x C
        
        elif m == 'MC-Dropout':
            p = 0.25

            if args.model == 'lenet':
                mc_net = model.LeNet5_Dropout(p=p).to(device)
            elif args.model == 'resnet9':
                mc_net = model.ResNet9(in_channels=n_channels, num_classes=n_output, p=p).to(device)
            elif args.model == 'wrn':
                mc_net = model.WRN(depth=28, widening_factor=5, num_classes = n_output, drop_rate=p).to(device)
            elif args.model == 'resnet50':
                mc_net = model.ResNet50(in_channels=n_channels, num_classes = n_output, p = p).to(device)
            mc_net.apply(df2.init_weights)
            mc_net.eval()
            num_weights = sum(p.numel() for p in mc_net.parameters() if p.requires_grad)

            if args.model == 'resnet50':
                optimizer = torch.optim.SGD(mc_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(mc_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if args.model == 'resnet9' and args.dataset == 'cifar':
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

            train_loss, train_acc, _, _ = df2.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                        model=mc_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                        scheduler=scheduler,epochs=epochs,
                                                                        verbose=False, train_mode=False, progress_bar=args.progress)
            train_res[m]['nll'].append(train_loss)
            train_res[m]['acc'].append(train_acc)
            
            T = 100
            id_mean, id_var, id_predictions = df2.dropout_sampler(dataset=test_data,model=mc_net,T=T,n_output=n_output,bs=batch_size) # id_predictions -> S x N x C
            ood_mean, ood_var, ood_predictions = df2.dropout_sampler(dataset=ood_test_data,model=mc_net,T=T,n_output=n_output,bs=batch_size) # ood_predictions -> S x N x C

        t2 = time.time()
        test_res[m]['time'].append(t2-t1)

        p_dist = Categorical(probs=id_mean.to(device))
        test_res[m]['nll'].append(-p_dist.log_prob(test_y.to(device)).mean().item())
        test_res[m]['acc'].append((id_mean.argmax(1).to(device) == test_y.to(device)).type(torch.float).mean().item())
        test_res[m]['ece'].append(metric(id_mean.to(device),test_y.to(device)).cpu().item())

        oodauc, aucroc = df2.auc_metric(id_mean, ood_mean, logits=False)
        test_res[m]['oodauc'].append(oodauc)
        test_res[m]['aucroc'].append(aucroc)

        ### print (train and) test prediction results
        print(f"\n--- Method {m} ---")
        if m in train_res:
            print("\nTrain Results:")
            print(f"Train Loss: {train_res[m]['nll'][ei]:.3f}; Train Acc: {train_res[m]['acc'][ei]:.1%}")
        print("\nTest Prediction:")
        t = time.strftime("%H:%M:%S", time.gmtime(test_res[m]['time'][ei]))
        print(f"Time h:m:s: {t}")
        print(f"Acc.: {test_res[m]['acc'][ei]:.1%}; ECE: {test_res[m]['ece'][ei]:.1%}; NLL: {test_res[m]['nll'][ei]:.3}")
        print(f"OOD-AUC: {test_res[m]['oodauc'][ei]:.1%}; AUC-ROC: {test_res[m]['aucroc'][ei]:.1%}")

        if m != 'MAP':
            # Save predictions
            prediction_dict[m] = {'id': id_predictions,
                                    'ood': ood_predictions}
            id_correct, id_incorrect = df2.sort_preds(id_predictions.to(device),torch.tensor(test_data.targets).to(device))

            prob_var_dict[m] = {'id_correct': id_correct.var(0),
                                'id_incorrect': id_incorrect.var(0),
                                    'ood': ood_predictions.var(0)}
            

    # Save variances for testing
    if args.save_var:
        torch.save(prob_var_dict,res_dir + f"prob_var_dict_{ei}.pt")
        torch.save(prediction_dict,res_dir + f"prediction_dict_{ei}.pt")

    res_violin_v = res_dir + f"vv_{ei}.pdf"

    df2.violin_var(var_dict=prob_var_dict, probit_sum = True, show=False, save_file=res_violin_v, title='Probit Variance', text = False)

## Record results
res_text = res_dir + f"result.txt"
results = open(res_text,'w')
torch.save(train_res,res_dir + f'train_res.pt')
torch.save(test_res,res_dir + f'test_res.pt')

print(f'train_res = {train_res}')
print(f'test_res = {test_res}')

percentage_metrics = ['acc','ece','oodauc','aucroc']

results.write(" --- MAP Training Details --- \n")
results.write(f"epochs: {epochs}; M: {M}; lr: {learning_rate}; weight_decay: {weight_decay}\n")

results.write("\n --- NUQLS Details --- \n")
results.write(f"epochs: {args.nuqls_epoch}; S: {M}; lr: {args.nuqls_lr}; weight_decay: {args.nuqls_wd}; init scale: {args.nuqls_scale}\n")

for m in methods:
    if args.n_experiment > 1:
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
                    print(f'list = {train_res[m][k][0]}')
                    results.write(f"{k}: {train_res[m][k][0]:.1%}\n")
                else:
                    print(f'list = {train_res[m][k][0]}')
                    results.write(f"{k}: {train_res[m][k][0]:.3f}\n")
        results.write("\n - Test Prediction: - \n")
        for k in test_res[m].keys():
            if k == 'time':
                t = time.strftime("%H:%M:%S", time.gmtime(test_res[m][k][0]))
                results.write(f"{k}: {t}\n")
            elif k in percentage_metrics:
                print(f'list = {test_res[m][k][0]}')
                results.write(f"{k}: {test_res[m][k][0]:.1%}\n")
            else:
                print(f'list = {test_res[m][k][0]}')
                results.write(f"{k}: {test_res[m][k][0]:.3f}\n")

results.close()