### --- Dependencies --- ###
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import scipy
from scipy.io import arff
import torch.nn.functional as F
from tqdm import tqdm
import posteriors.nuqls as nuqls
from posteriors.lla.models import MLPS
from posteriors.lla.likelihoods import GaussianLh
from posteriors.lla.laplace import Laplace
import argparse
import time
import os

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Regression Experiment')
parser.add_argument('--dataset', required=True,type=str,help='Dataset')
parser.add_argument('--n_experiment',default=10,type=int,help='number of experiments')
parser.add_argument('--activation',default='tanh',type=str,help='Non-linear activation for MLP')
parser.add_argument('--nuqls_epoch',default=1000,type=int,help='epochs for nuqls.')
parser.add_argument('--nuqls_S',default=100,type=int,help='num realisations for nuqls.')
parser.add_argument('--nuqls_lr',default=5e-4,type=float,help='lr for nuqls.')
args = parser.parse_args()

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n Using {device} device")

### --- INPUT DATA HERE AS WELL AS DATASET NAME --- ###

if args.dataset == 'energy':
    df = pd.read_excel('./data/energy/ENB2012_data.xlsx')
    input_dim = 8
    hidden_sizes = [150]
    epochs = 1500
    de_epochs = 1500
elif args.dataset == 'concrete':
    df = pd.read_excel('./data/concrete/Concrete_Data.xls')
    input_dim = 8
    hidden_sizes = [150]
    epochs = 1500
    de_epochs = 300
elif args.dataset == 'kin8nm':
    arff_file = arff.loadarff('./data/kin8mn/dataset_2175_kin8nm.arff')
    df = pd.DataFrame(arff_file[0])
    input_dim = 8
    hidden_sizes = [100,100]
    epochs = 500
    de_epochs = 100

# print(data.shape)
print("--- Loading dataset {} --- \n".format(args.dataset))
print("Number of data points = {}".format(len(df)))
print("Number of coloumns = {}".format(len(df.columns)))
print("Number of features = {}".format(input_dim))

## Calibration function
def calibration_curve_r(loader,mean,variance,c):
    predicted_conf = torch.linspace(0,1,c)
    observed_conf = torch.empty((c))
    for i,ci in enumerate(predicted_conf):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_l = mean.reshape(-1) - z*torch.sqrt(variance.reshape(-1))
        ci_r = mean.reshape(-1) + z*torch.sqrt(variance.reshape(-1)) 
        correct = 0
        for iy,(_,y) in enumerate(loader):
            y = y.to(device)
            if ci_l[iy] < y.reshape(-1) and y.reshape(-1) < ci_r[iy]:
                correct += 1
        observed_conf[i] = correct / len(loader)
    return observed_conf,predicted_conf

class RegressionDataset(Dataset):
    '''
    Prepare dataset for regression.
    Input the number of features.

    Input:
    - dataset: numpy array

    Returns:
        - Tuple (X,y) - X is a numpy array, y is a double value.
    '''
    def __init__(self, dataset, input_dim, mX=0, sX=1, my=0, sy=1):
        self.X, self.y = dataset[:,:input_dim], dataset[:,input_dim]
        self.X, self.y = (self.X - mX)/sX, (self.y - my)/sy
        self.len_data = self.X.shape[0]

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        return self.X[i,:], self.y[i]
        
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight,mean=0,std=1)
        nn.init.normal_(m.bias,mean=0,std=1)

def train(dataloader, model, optimizer, loss_function, scheduler=None):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Get and prepare inputs
        y = y.reshape(-1,1)
        
        # Perform forward pass
        pred = model(X)
        
        # Compute loss
        loss = loss_function(pred, y)
        
        # Perform backward pass
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss = train_loss / (i+1)
    return train_loss

def test(dataloader, model, my, sy, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y = y.reshape((y.shape[0],1))
            pred = model(X)
            pred = pred * sy + my
            test_loss += loss_function(pred, y).item()
    test_loss /= (i+1)
    return test_loss

def to_np(x):
    return x.cpu().detach().numpy()

class EnsembleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim,hidden_sizes[0])
        if len(hidden_sizes) > 1:
            self.linear_2 = nn.Linear(hidden_sizes[1],hidden_sizes[1])
        if args.activation == 'tanh':
            self.act = nn.Tanh()
        elif args.activation == 'relu':
            self.act = nn.ReLU()
        self.linear_mu = nn.Linear(hidden_sizes[-1],1)
        self.linear_sig = nn.Linear(hidden_sizes[-1],1)

    def forward(self, x):
        x = self.act(self.linear_1(x))
        if len(hidden_sizes) > 1:
            x = self.act(self.linear_2(x))
        mu = self.linear_mu(x)
        variance = self.linear_sig(x)
        variance = F.softplus(variance) + 1e-6
        return mu, variance

class CustomNLL(nn.Module):
    def __init__(self):
        super(CustomNLL, self).__init__()

    def forward(self, y, mean, var):
        
        loss = (0.5*torch.log(var) + 0.5*(y - mean).pow(2)/var).mean() + 1

        if np.any(np.isnan(to_np(loss))):
            print(torch.log(var))
            print((y - mean).pow(2)/var)
            raise ValueError('There is Nan in loss')
        
        return loss

def train_de(dataloader, model, optimizer, loss_function, scheduler=None):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        y = y.reshape((-1,1))
        pred, var = model(X)
        loss = loss_function(pred, y, var)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss = train_loss / (i+1)
    return train_loss

def test_de(dataloader, model, my, sy, loss_function, mse_loss):
    model.eval()
    test_loss = 0
    mse = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y = y.reshape((-1,1))
            mean, variance = model(X)
            mean = mean * sy + my
            variance = variance * (sy**2)
            test_loss += loss_function(mean, y, variance).item()
            mse += mse_loss(mean,y).item()
    test_loss /= (i+1)
    mse /= (i+1)
    return test_loss, mse

train_ratio = 0.7
normalize = True

## Num of points and dimension of data
num_points = len(df)
dimension = len(df.columns)-1

train_size = int(num_points*train_ratio)
validation_size = int(num_points*((1-train_ratio)/2))
test_size = int(num_points - train_size - validation_size)
dataset_numpy = df.values

if normalize:
    mx = dataset_numpy[:,:input_dim].mean(0)
    my = dataset_numpy[:,input_dim].mean(0)
    sx = dataset_numpy[:,:input_dim].std(0)
    sy = dataset_numpy[:,input_dim].std(0)

# Training parameters

lr = 1e-3
weight_decay = 1e-5
batch_size = 100
mse_loss = nn.MSELoss(reduction='mean')
nll = torch.nn.GaussianNLLLoss()
M = 10

# Setup metrics
methods = ['MAP','NUQLs','DE','LLA']
train_methods = ['MAP','NUQLs','DE']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'loss': []}
    test_res[m] = {'rmse': [],
                  'nll': [],
                  'ece': [],
                  'time': []}


for ei in tqdm(range(args.n_experiment)):
    print("\n--- experiment {} ---".format(ei))
    np.random.shuffle(dataset_numpy)
    training_set, validation_set, test_set = dataset_numpy[:train_size,:], dataset_numpy[train_size:train_size+validation_size], dataset_numpy[train_size+validation_size:,:]

    train_dataset = RegressionDataset(training_set, input_dim=input_dim, mX=mx, sX=sx, my=my, sy=sy)
    validation_dataset = RegressionDataset(validation_set, input_dim=input_dim, mX=mx, sX=sx, my=my, sy=sy)
    test_dataset = RegressionDataset(test_set, input_dim=input_dim, mX=mx, sX=sx, my=my, sy=sy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
    test_x, test_y = next(iter(test_loader))
    calibration_test_loader = DataLoader(test_dataset,1)
    calibration_test_loader_val = DataLoader(validation_dataset,1)

    for m in methods:
        t1 = time.time()
        if m == 'MAP':
            map_net = MLPS(input_size=input_dim, hidden_sizes=hidden_sizes, output_size=1, activation=args.activation, flatten=False, bias=True).to(device=device, dtype=torch.float64)
            map_net.apply(weights_init)
            map_p = sum(p.numel() for p in map_net.parameters() if p.requires_grad)

            adam_optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.PolynomialLR(adam_optimizer, total_iters=epochs*10, power=0.5)

            map_t1 = time.time()
            # Run the training loop
            for epoch in tqdm(range(epochs)):
                map_train_loss  = train(train_loader, map_net, optimizer=adam_optimizer, loss_function=mse_loss, scheduler=scheduler)
                map_test_loss = test(test_loader, map_net, my=0, sy=1, loss_function=mse_loss)
            train_res['MAP']['loss'].append(map_train_loss)

        elif m == 'NUQLs':
            best_scale = 1e17
            best_ece = 1.01
            scales = torch.linspace(0,1,20)

            for s,scale in enumerate(scales):
                nuqls_model = nuqls.linear_nuqls_small(net = map_net, train = train_dataset, S = args.nuqls_S, epochs=args.nuqls_epoch, lr=args.nuqls_lr, bs=train_size, bs_test=test_size,init_scale=scale)
                new_nuqls_predictions,_,_ = nuqls_model.method(validation_dataset,mu=0.9,weight_decay=0,my=0,sy=1,verbose=False)
                new_observed_conf_nuqls, new_predicted_conf = calibration_curve_r(calibration_test_loader_val,new_nuqls_predictions.mean(1),new_nuqls_predictions.var(1),11)
                new_ece_nuqls = torch.mean(torch.square(new_observed_conf_nuqls - new_predicted_conf))
                if new_ece_nuqls < best_ece:
                    best_ece = new_ece_nuqls
                    nuqls_predictions = new_nuqls_predictions
                    best_scale = scale
                    # print("New best ece = {}".format(new_ece_nuqls))
                    # print("New best scale = {}".format(best_scale))
                else:
                    break

            nuqls_model = nuqls.linear_nuqls_small(net = map_net, train = train_dataset, S = args.nuqls_S, epochs=args.nuqls_epoch, lr=args.nuqls_lr, bs=train_size, bs_test=test_size,init_scale=best_scale)
            nuqls_predictions, max_l2_loss, norm_resid = nuqls_model.method(test_dataset,mu=0.9,weight_decay=0,my=0,sy=1,verbose=False)

            mean_pred = nuqls_predictions.mean(1)
            var_pred = nuqls_predictions.var(1)

            train_res['NUQLs']['loss'].append(max_l2_loss.detach().cpu().item())

        elif m == 'DE':
            model_list = []
            opt_list = []
            sched_list = []

            for i in range(M):
                model_list.append(EnsembleNetwork().to(device=device,dtype=torch.float64))
                model_list[i].apply(weights_init)
                opt_list.append(torch.optim.Adam(model_list[i].parameters(), lr = lr, weight_decay = weight_decay))
                sched_list.append(torch.optim.lr_scheduler.PolynomialLR(opt_list[i], total_iters=de_epochs*10, power=0.5))

            de_train_total = 0
            de_test_total = 0
            de_mse_total = 0
            de_t1 = time.time()
            for i in range(M):
                for epoch in tqdm(range(de_epochs)):
                    de_train_loss = train_de(dataloader=train_loader, model=model_list[i], optimizer=opt_list[i], loss_function=nll, scheduler=None)
                    de_test_loss, de_test_mse = test_de(test_loader, model=model_list[i], my=0, sy=1, loss_function=nll, mse_loss=mse_loss)
                de_train_total += de_train_loss
                de_test_total += de_test_loss
                de_mse_total += de_test_mse
            de_train_total /= M
            de_test_total /= M
            de_mse_total /= M

            train_res['DE']['loss'].append(de_train_total)

            ensemble_het_mu = torch.empty((M,test_size))
            ensemble_het_var = torch.empty((M,test_size))
            for i in range(M):
                for X,y in test_loader:
                    X,y = X.to(device), y.to(device)
                    y = y.reshape((y.shape[0],1))
                    mu, sig = model_list[i](X)
                    mu = mu
                    sig = sig
                    ensemble_het_mu[i,:] = mu.reshape(1,-1)
                    ensemble_het_var[i,:] = sig.reshape(1,-1)

            mean_pred = torch.mean(ensemble_het_mu,dim=0)
            var_pred = torch.mean(ensemble_het_var + torch.square(ensemble_het_mu), dim=0) - torch.square(mean_pred)

        elif m == 'LLA':
            full_train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=False)
            full_test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
            X_test, Y_test = next(iter(full_test_loader))
            X_test = X_test.to(device)

            full_val_loader = DataLoader(validation_dataset,batch_size=len(validation_dataset),shuffle=False)
            X_val, Y_val = next(iter(full_val_loader))
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            best_sigma = 1e-17
            best_ll = -1e17
            sigmas = torch.logspace(-2,2,10)
            for sigma in sigmas:
                lh = GaussianLh(sigma_noise=sigma.to(device))
                new_ll = lh.log_likelihood(Y_val.to(device),map_net(X_val))
                if new_ll > best_ll:
                    best_ll = new_ll
                    best_sigma = sigma
                    # print("New best ll = {}".format(best_ll))
                    # print("New best sigma = {}".format(best_sigma))
            lh = GaussianLh(sigma_noise=best_sigma)

            best_prior = 1e-17
            best_ece = 1.01
            priors = torch.logspace(-2,3,10)

            ## Find prior precision on validation set
            for s,prior in enumerate(priors):

                lap = Laplace(map_net, float(prior), lh)
                lap.infer(full_train_loader, cov_type='full', dampen_kron=True)
                new_lla_mu, new_lla_var = lap.predictive_samples_glm(X_val, n_samples=1000)
                new_lla_mu, new_lla_var = new_lla_mu.detach(), new_lla_var.detach()

                new_observed_conf_lla, new_predicted_conf = calibration_curve_r(calibration_test_loader_val,new_lla_mu,new_lla_var,11)
                new_ece_lla = torch.mean(torch.square(new_observed_conf_lla - new_predicted_conf))
                if new_ece_lla < best_ece:
                    best_ece = new_ece_lla
                    lla_mu = new_lla_mu
                    lla_var = new_lla_var
                    best_prior = prior
                    # print("New best ece = {}".format(new_ece_lla))
                    # print("New best prior = {}".format(best_prior))

            lap = Laplace(map_net, float(best_prior), lh)
            lap.infer(full_train_loader, cov_type='full', dampen_kron=True)
            lla_mu, lla_var = lap.predictive_samples_glm(X_test, n_samples=1000)
            mean_pred, var_pred = lla_mu.detach(), lla_var.detach()

        print(f"\n--- Method {m} ---")
        if m in train_res:
            print("\nTrain Results:")
            print(f"Train Loss: {train_res[m]['loss'][ei]:.3f}")

        if m != 'MAP':
            t2 = time.time()
            test_res[m]['time'].append(t2-t1)

            test_res[m]['rmse'].append(torch.sqrt(mse_loss(mean_pred.detach().cpu().reshape(-1,1),test_y.reshape(-1,1))).detach().cpu().item())

            test_res[m]['nll'].append(nll(mean_pred.detach().cpu().reshape(-1,1),test_y.reshape(-1,1),var_pred.detach().cpu().reshape(-1,1)).detach().cpu().item())

            observed_conf, predicted_conf = calibration_curve_r(calibration_test_loader,mean_pred,var_pred,11)
            test_res[m]['ece'].append(torch.mean(torch.square(observed_conf - predicted_conf)).detach().cpu().item())

            
            print("\nTest Prediction:")
            t = time.strftime("%H:%M:%S", time.gmtime(test_res[m]['time'][ei]))
            print(f"Time h:m:s: {t}")
            print(f"RMSE.: {test_res[m]['rmse'][ei]:.3f}; NLL: {test_res[m]['nll'][ei]:.3f}; ECE: {test_res[m]['ece'][ei]:.1%}")
        print('\n')

## Record results
res_dir = "./results/uci_regression"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

results = open(f"{res_dir}/{args.dataset}.txt",'w')

results.write("Training, Val, Test points = {}, {}, {}\n".format(train_size, validation_size, test_size))
results.write(f"Number of hidden units, parameters = {hidden_sizes}, {map_p}\n")

results.write("\n--- MAP Training Details --- \n")
results.write("training: epochs, de_epochs, lr, weight decay, batch size = {}, {}, {}, {}\n".format(
    epochs, de_epochs, lr, weight_decay, batch_size
))

results.write("\n --- NUQLs Details --- \n")
results.write(f"epochs: {args.nuqls_epoch}; S: {M}; lr: {args.nuqls_lr}; epochs: {args.nuqls_epoch}\n")

for m in methods:
    if args.n_experiment > 1:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                results.write(f"{k}: {np.mean(train_res[m][k]):.3f} +- {np.std(train_res[m][k]):.3f} \n")
        if m != 'MAP':
            results.write("\n - Test Prediction: - \n")
            for k in test_res[m].keys():
                if k == 'time':
                    t = time.strftime("%H:%M:%S", time.gmtime(np.mean(test_res[m][k])))
                    results.write(f"{k}: {t}\n")
                else:
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
                if k == 'time':
                    t = time.strftime("%H:%M:%S", time.gmtime(test_res[m][k][0]))
                    results.write(f"{k}: {t}\n")
                else:
                    print(f'list = {test_res[m][k][0]}')
                    results.write(f"{k}: {test_res[m][k][0]:.3f}\n")
results.close()
