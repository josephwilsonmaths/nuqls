### --- Dependencies --- ###
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import posteriors.nuqls as nuqls
from cuqls.posterior import Cuqls
from posteriors.lla.models import MLPS
from posteriors.lla.likelihoods import GaussianLh
from posteriors.lla.laplace import Laplace
import posteriors.lla_s as lla_s
from posteriors.de import DeepEnsemble
import argparse
import time
import os
import configparser
import utils.datasets
import json
import posteriors.util as util
import posteriors.swag as swag
import utils.regression_util as utility
# import posteriors.cuqls as cuqls

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Regression Experiment')
# parser.add_argument('--dataset', required=True,type=str,help='Relative path to list of datasets. Each dataset should be placed on new line with no whitespace around it.')
parser.add_argument('--dataset', required=True,type=str,help='Dataset')
parser.add_argument('--n_experiment',default=10,type=int,help='number of experiments')
parser.add_argument('--activation',default='tanh',type=str,help='Non-linear activation for MLP')
parser.add_argument('--verbose', action='store_true',help='verbose flag for results')
parser.add_argument('--extra_verbose', action='store_true',help='verbose flag for training')
parser.add_argument('--progress_bar', action='store_true',help='progress bar flag for all methods')
args = parser.parse_args()

# # Parse datasets
# r = open(args.dataset,"r")
# datasets = []
# for d in r:
#     datasets.append(d.strip())

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n Using {device} device")

# # Iterate through datasets:
# for dataset in datasets:

# Get hyperparameters from config file
config = configparser.ConfigParser()
config.read('utils/regression.ini')
df = utils.datasets.read_regression(args.dataset)
n_experiment = config.getint(args.dataset,'n_experiment')
input_start = config.getint(args.dataset,'input_start')
input_dim = config.getint(args.dataset,'input_dim')
target_dim = config.getint(args.dataset,'target_dim')
hidden_sizes = json.loads(config.get(args.dataset,'hidden_sizes'))
epochs = config.getint(args.dataset,'epochs')
lr = config.getfloat(args.dataset,'lr')
de_epochs = config.getint(args.dataset,'de_epochs')
de_lr = config.getfloat(args.dataset,'de_lr')
weight_decay = config.getfloat(args.dataset,'weight_decay')
nuqls_S = config.getint(args.dataset,'nuqls_S')
nuqls_epoch = config.getint(args.dataset,'nuqls_epoch')
nuqls_lr = config.getfloat(args.dataset,'nuqls_lr')

# Fixed parameters
train_ratio = 0.7
normalize = True
batch_size = 200
mse_loss = nn.MSELoss(reduction='mean')
nll = torch.nn.GaussianNLLLoss()
S = 10

# Give dataframe summary
print("--- Loading dataset {} --- \n".format(args.dataset))
print("Number of data points = {}".format(len(df)))
print("Number of coloumns = {}".format(len(df.columns)))
print("Number of features = {}".format(input_dim-input_start))

## Num of points and dimension of data
num_points = len(df)
dimension = len(df.columns)-1

train_size = int(num_points*train_ratio)
validation_size = int(num_points*((1-train_ratio)/2))
test_size = int(num_points - train_size - validation_size)
dataset_numpy = df.values

# Normalize the dataset
if normalize:
    mx = dataset_numpy[:,input_start:input_dim].mean(0)
    my = dataset_numpy[:,target_dim].mean(0)
    sx = dataset_numpy[:,input_start:input_dim].std(0)
    sx = np.where(sx==0,1,sx)
    sy = dataset_numpy[:,target_dim].std(0)
    sy = np.where(sy==0,1,sy)

# Setup metrics
# methods = ['MAP','NUQLS_SCALE_S','NUQLS_SCALE_S_MAP','DE','LLA','SWAG'] 
# methods = ['MAP','NUQLS','NUQLS_MAP','NUQLS_SCALE_S','NUQLS_SCALE_S_MAP','NUQLS_SCALE_T','NUQLS_SCALE_T_MAP','DE','LLA','SWAG']
# methods = ['MAP','NUQLS_SCALE_S','NUQLS_SCALE_S_MAP','DE','LLA','SWAG']
methods = ['MAP', 'CUQLS']
train_methods = ['MAP','CUQLS','DE']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'loss': [],
                        'time': []}
    test_res[m] = {'rmse': [],
                'nll': [],
                'ece': [],
                'time': []}

# Iterate through number of experiments
for ei in tqdm(range(n_experiment)):
    print("\n--- experiment {} ---".format(ei))
    np.random.shuffle(dataset_numpy) # Randomness
    training_set, validation_set, test_set = dataset_numpy[:train_size,:], dataset_numpy[train_size:train_size+validation_size], dataset_numpy[train_size+validation_size:,:]

    train_dataset = utils.datasets.RegressionDataset(training_set, 
                                    input_start=input_start, input_dim=input_dim, target_dim=target_dim,
                                    mX=mx, sX=sx, my=my, sy=sy)
    validation_dataset = utils.datasets.RegressionDataset(validation_set, 
                                    input_start=input_start, input_dim=input_dim, target_dim=target_dim,
                                    mX=mx, sX=sx, my=my, sy=sy)
    test_dataset = utils.datasets.RegressionDataset(test_set, 
                                    input_start=input_start, input_dim=input_dim, target_dim=target_dim,
                                    mX=mx, sX=sx, my=my, sy=sy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
    _, test_y = next(iter(test_loader))

    calibration_test_loader = DataLoader(test_dataset,1)
    calibration_test_loader_val = DataLoader(validation_dataset,len(validation_dataset))
    _, val_y = next(iter(calibration_test_loader_val))

    for m in methods:
        print(f'METHOD:: {m}')
        t1 = time.time()
        if m == 'MAP':
            map_net = MLPS(input_size=input_dim-input_start, hidden_sizes=hidden_sizes, output_size=1, activation=args.activation, flatten=False, bias=True).to(device=device, dtype=torch.float64)
            
            # map_net = ResRegressionDNN().to(device=device,dtype=torch.float64)
            map_net.apply(utility.weights_init)
            map_p = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            print(f'parameters of network = {map_p}')
            
            if args.dataset=='kin8nm' or args.dataset=='wine' or args.dataset=='naval' or args.dataset=='protein' or args.dataset=='song':
                optimizer = torch.optim.SGD(map_net.parameters(),lr=lr, weight_decay=weight_decay, momentum=0.9)
                scheduler = None
            else:
                optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs*10, power=0.5)

            if args.verbose:
                full_train_loader = DataLoader(train_dataset, len(train_dataset))
                train_x, _  = next(iter(full_train_loader))
                ntk_rank, ntk_linear_dependence, ntk_shape, ck_rank, ck_linear_dependence, ck_shape = utility.rank_calc(map_net, train_x)
                print(f'Before training: NTK ({ntk_shape}) | rank: {ntk_rank}, nullity: {ntk_shape[1]-ntk_rank}, lin_dep: {ntk_linear_dependence:.3f}, CK ({ck_shape}) | rank: {ck_rank}, nullity: {ck_shape[1]-ck_rank}, lin_dep: {ck_linear_dependence}')

            # Run the training loop
            for epoch in tqdm(range(epochs)):
                map_train_loss  = utility.train(train_loader, map_net, optimizer=optimizer, loss_function=mse_loss, scheduler=scheduler)
                map_test_loss = utility.test(test_loader, map_net, my=0, sy=1, loss_function=mse_loss)
                if args.extra_verbose and epoch % 10 == 0:
                    print("Epoch {} of {}".format(epoch,epochs))
                    print("Training loss = {:.4f}".format(map_train_loss))
                    print("Test loss = {:.4f}".format(map_test_loss))
                    print("\n -------------------------------------")
            train_res[m]['loss'].append(map_train_loss)

            print(f'MAP: TEST RMSE = {np.sqrt(map_test_loss):.4}')

            if args.verbose:
                full_train_loader = DataLoader(train_dataset, len(train_dataset))
                train_x, _  = next(iter(full_train_loader))
                ntk_rank, ntk_linear_dependence, ntk_shape, ck_rank, ck_linear_dependence, ck_shape = utility.rank_calc(map_net, train_x)
                print(f'After training: NTK ({ntk_shape}) | rank: {ntk_rank}, nullity: {ntk_shape[1]-ntk_rank}, lin_dep: {ntk_linear_dependence:.3f}, CK ({ck_shape}) | rank: {ck_rank}, nullity: {ck_shape[1]-ck_rank}, lin_dep: {ck_linear_dependence}')

            map_pred = []
            for x,_ in test_loader:
                x = x.to(device)
                map_pred.append(map_net(x))
            map_test_pred = torch.cat(map_pred)

            map_pred = []
            val_loader = DataLoader(validation_dataset, batch_size=test_size, shuffle=False)
            for x,_ in val_loader:
                x = x.to(device)
                map_pred.append(map_net(x))
            map_val_pred = torch.cat(map_pred)

        elif m == 'CUQLS':
            cuqls_posterior = Cuqls(network=map_net, task='regression')
            loss,resid = cuqls_posterior.train(train=train_dataset,
                                    batchsize=batch_size, 
                                    scale=0.01, 
                                    S=nuqls_S, 
                                    epochs=nuqls_epoch, 
                                    lr=nuqls_lr, 
                                    mu=0.9, 
                                    verbose=True)
            train_res[m]['loss'].append(loss)
            cuqls_posterior.HyperparameterTuning(validation_dataset, 0.01, 10000, 100, verbose=args.extra_verbose)

            cuqls_test_preds = cuqls_posterior.test(test_dataset, test_bs=batch_size)
            scaled_test_preds = cuqls_test_preds*cuqls_posterior.scale_cal
            print(f'Scale cal: {cuqls_posterior.scale_cal:.4}')

            mean_pred = cuqls_test_preds.mean(1)
            var_pred = scaled_test_preds.var(1)

        elif m == 'DE':
            de_network = utility.EnsembleNetwork(hidden_sizes,input_start,input_dim,args.activation).to(device=device,dtype=torch.float64)
            de_posterior = DeepEnsemble(network=de_network, task='regression', M = 10)
            train_nll, train_mse = de_posterior.train(loader=train_loader, 
                                                    lr=de_lr, 
                                                    wd=weight_decay,
                                                    epochs=de_epochs, 
                                                    optim_name='adam', 
                                                    sched_name=None, 
                                                    verbose=True,
                                                    extra_verbose=True)
            train_res[m]['loss'].append(train_nll)
            mean_pred, var_pred = de_posterior.test(test_loader)

        elif m == 'LLA':
            if args.dataset == 'protein':
                cov_type = 'kron'
            elif args.dataset == 'song':
                cov_type = 'diag'
            else:
                cov_type = 'full'

            full_train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=False)
            full_test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

            full_val_loader = DataLoader(validation_dataset,batch_size=len(validation_dataset),shuffle=False)
            X_val, Y_val = next(iter(full_val_loader))
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            lla_val_loader = DataLoader(validation_dataset, batch_size=batch_size)
            lla_test_loader = DataLoader(test_dataset, batch_size=batch_size)

            best_sigma = 1e-17
            best_ll = -1e17
            sigmas = torch.logspace(-2,2,10)
            for sigma in sigmas:
                lh = GaussianLh(sigma_noise=sigma.to(device))
                new_ll = lh.log_likelihood(Y_val.to(device),map_net(X_val))

                # Print info
                if args.verbose:
                    print(f'\nNOISE {sigma:.3}:: LLA: {new_ll:.4f}') 

                if new_ll > best_ll:
                    best_ll = new_ll
                    best_sigma = sigma
                    if args.verbose:
                        print('New best!')
            lh = GaussianLh(sigma_noise=best_sigma)

            best_prior = 1e-17
            best_ece = 1.01
            priors = torch.logspace(-2,3,10)

            ## Find prior precision on validation set
            for s,prior in enumerate(priors):

                lap = Laplace(map_net, float(prior), lh)
                lap.infer(train_loader, cov_type=cov_type, dampen_kron=False)
                
                new_lla_mu = []
                new_lla_var = []
                for x,_ in lla_val_loader:
                    mu, var = lap.predictive_samples_glm(x.to(device), n_samples=1000)
                    new_lla_mu.append(mu.detach())
                    new_lla_var.append(var.detach())
                new_lla_mu = torch.cat(new_lla_mu,dim=0)
                new_lla_var = torch.cat(new_lla_var,dim=0)

                new_observed_conf_lla, new_predicted_conf = utility.calibration_curve_r(val_y,new_lla_mu,new_lla_var,11)
                new_ece_lla = torch.mean(torch.square(new_observed_conf_lla - new_predicted_conf))

                # Print info
                if args.verbose:
                    print(f'\nSCALE {prior:.3}:: LLA: {new_ece_lla:.1%}') 
                
                if new_ece_lla < best_ece:
                    best_ece = new_ece_lla
                    lla_mu = new_lla_mu
                    lla_var = new_lla_var
                    best_prior = prior
                    if args.verbose:
                        print("New best!")

            lap = Laplace(map_net, float(best_prior), lh)
            lap.infer(train_loader, cov_type=cov_type, dampen_kron=False)

            lla_mu = []
            lla_var = []
            for x,_ in lla_test_loader:
                mu, var = lap.predictive_samples_glm(x.to(device), n_samples=1000)
                lla_mu.append(mu.detach())
                lla_var.append(var.detach())
            mean_pred = torch.cat(lla_mu,dim=0)
            var_pred = torch.cat(lla_var,dim=0)

        elif m=='SWAG':
            wds = torch.tensor([0,5e-3,1e-5])
            best_ece = 1.01
            for wd in wds:
                ## SWAG
                swag_method = swag.SWAG_R(map_net,epochs = epochs, lr = lr, cov_mat = True,
                                            max_num_models=10)
                swag_method.train_swag(train_loader=train_loader, weight_decay=wd)

                T = 1000
                swag_pred = []
                for t in range(T):
                    swag_method.sample(cov=True)
                    pred_i = util.evaluate_batch(dataset=validation_dataset, model=swag_method, batch_size=batch_size)
                    swag_pred.append(pred_i.reshape(1,-1))
                swag_pred = torch.cat(swag_pred)
                mean_pred = swag_pred.mean(axis=0)
                var_pred = swag_pred.var(axis=0)

                conf, predicted = utility.calibration_curve_r(val_y,mean_pred,var_pred,11)
                new_ece_swag = torch.mean(torch.square(conf - predicted))

                if new_ece_swag < best_ece:
                    best_ece = new_ece_swag
                    best_wd = wd
                    if args.verbose:
                        print("New best!")

            ## SWAG
            swag_method = swag.SWAG_R(map_net,epochs = epochs, lr = lr, cov_mat = True,
                                        max_num_models=10)
            swag_method.train_swag(train_loader=train_loader, weight_decay=best_wd)

            T = 1000
            swag_pred = []
            for t in range(T):
                swag_method.sample(cov=True)
                pred_i = util.evaluate_batch(dataset=test_dataset, model=swag_method, batch_size=batch_size)
                swag_pred.append(pred_i.reshape(1,-1))
            swag_pred = torch.cat(swag_pred)
            mean_pred = swag_pred.mean(axis=0)
            var_pred = swag_pred.var(axis=0)

        print(f"\n--- Method {m} ---")
        t2 = time.time()
        if m in train_res:
            print("\nTrain Results:")
            print(f"Train Loss: {train_res[m]['loss'][ei]:.3f}")
            # if m == 'MAP':
            train_res[m]['time'].append(t2-t1)
            t = train_res[m]['time'][ei]
            print(f'Time(s): {t:.3f}')

        if m != 'MAP':
            test_res[m]['time'].append(t2-t1)

            test_res[m]['rmse'].append(torch.sqrt(mse_loss(mean_pred.detach().cpu().reshape(-1,1),test_y.reshape(-1,1))).detach().cpu().item())

            test_res[m]['nll'].append(nll(mean_pred.detach().cpu().reshape(-1,1),test_y.reshape(-1,1),var_pred.detach().cpu().reshape(-1,1)).detach().cpu().item())

            observed_conf, predicted_conf = utility.calibration_curve_r(test_y,mean_pred,var_pred,11)
            test_res[m]['ece'].append(torch.mean(torch.square(observed_conf - predicted_conf)).detach().cpu().item())

            print("\nTest Prediction:")
            # t = time.strftime("%H:%M:%S", time.gmtime(test_res[m]['time'][ei]))
            # print(f"Time h:m:s: {t}")
            t = test_res[m]['time'][ei]
            print(f'Time(s): {t:.3f}')
            print(f"RMSE.: {test_res[m]['rmse'][ei]:.3f}; NLL: {test_res[m]['nll'][ei]:.3f}; ECE: {test_res[m]['ece'][ei]:.1%}")
        print('\n')

## Record results
res_dir = "./results/uci_regression"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

results = open(f"{res_dir}/{args.dataset}.txt",'w')
results.write("Training, Val, Test points = {}, {}, {}\n".format(train_size, validation_size, test_size))
results.write(f"Number of hidden units, parameters = {hidden_sizes}, {map_p}\n")
results.write(f'n_experiment = {n_experiment}\n')

results.write("\n--- MAP Training Details --- \n")
results.write("training: epochs, de_epochs, lr, weight decay, batch size = {}, {}, {}, {}, {}\n".format(
    epochs, de_epochs, lr, weight_decay, batch_size
))

results.write("\n --- NUQLS Details --- \n")
results.write(f"epochs: {nuqls_epoch}; S: {nuqls_S}; lr: {nuqls_lr}\n")

for m in methods:
    if n_experiment > 1:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                results.write(f"{k}: {np.mean(train_res[m][k]):.3f} +- {np.std(train_res[m][k]):.3f} \n")
        if m != 'MAP':
            results.write("\n - Test Prediction: - \n")
            for k in test_res[m].keys():
                # if k == 'time':
                #     t = time.strftime("%H:%M:%S", time.gmtime(np.mean(test_res[m][k])))
                #     results.write(f"{k}: {t}\n")
                # else:
                results.write(f"{k}: {np.mean(test_res[m][k]):.3f} +- {np.std(test_res[m][k]):.3f} \n")
    else:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                results.write(f"{k}: {train_res[m][k][0]:.3f}\n")
        if m != 'MAP':
            results.write("\n - Test Prediction: - \n")
            for k in test_res[m].keys():
                # if k == 'time':
                #     t = time.strftime("%H:%M:%S", time.gmtime(test_res[m][k][0]))
                #     results.write(f"{k}: {t}\n")
                # else:
                results.write(f"{k}: {test_res[m][k][0]:.3f}\n")
results.close()

print('results created')

