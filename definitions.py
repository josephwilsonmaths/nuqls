### --- Dependencies --- ###
import torch
import importlib
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import datetime
import os
import solvers as solv
from importlib import reload
import random

reload(solv)

importlib.reload(solv)
from scipy.sparse.linalg import LinearOperator, lsmr
from tqdm import tqdm
from torch.func import vmap, jacrev
from functorch import make_functional, make_functional_with_buffers

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_dtype(torch.float64)

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
        self.X, self.y = torch.from_numpy(self.X), torch.from_numpy(self.y)

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        return self.X[i,:], self.y[i]
    
def data_split(df, train_ratio):
    num_points = len(df)
    train_size = int(num_points*train_ratio)
    dataset_numpy = df.values
    np.random.shuffle(dataset_numpy)
    training_set, test_set = dataset_numpy[:train_size,:], dataset_numpy[train_size:,:]
    print("training set has shape {} \n".format(training_set.shape))
    print("test set has shape {}".format(test_set.shape))
    return training_set, test_set

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_d, width):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_d, width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width,width),
      nn.ReLU(),
      nn.Linear(width, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)
  
class EnsembleNetwork(nn.Module):
    def __init__(self, input_d, width):
        super().__init__()
        self.linear_1 = nn.Linear(input_d,width)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(width,width)
        self.linear_mu = nn.Linear(width,1)
        self.linear_sig = nn.Linear(width,1)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) #Relu applied to input layer
        x = self.relu(self.linear_2(x)) #Relu applied to first hidden layer
        x = self.relu(self.linear_2(x)) #Relu applied to second hidden layer
        x = self.relu(self.linear_2(x)) #Relu applied to third hidden layer
        x = self.relu(self.linear_2(x)) #Relu applied to fourth hidden layer
        mu = self.linear_mu(x)
        variance = self.linear_sig(x)
        variance = F.softplus(variance) + 1e-6
        return mu, variance

def to_np(x):
    return x.cpu().detach().numpy()

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
  
def weights_init(m):
    p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight,mean=0,std=1/p)
        nn.init.normal_(m.bias,mean=0,std=0.01)
        # nn.init.xavier_normal_(m.bias)

def optimizer_shared(model, type='adam', learning_rate=1e-1):
    if type=='adam':
        return torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif type=='sgd':
        return torch.optim.SGD(model.parameters(), lr = learning_rate)
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def flatten_extend_gradient(parameters):
    flat_list = []
    for parameter in parameters:
        flat_list.extend(parameter.grad.detach().numpy().flatten())
    return flat_list

def gradient_model(model,optimizer,xi):
    ## model needs to have parameters with requires_grad=true
    optimizer.zero_grad()
    model(xi).backward()
    grad_vec = np.array(flatten_extend_gradient(list(model.parameters())))
    return grad_vec

def ntk_single(x1,x2,model, optimizer):
    j1 = gradient_model(model=model,optimizer=optimizer,xi=x1)
    j2 = gradient_model(model=model,optimizer=optimizer,xi=x2)
    return j1 @ j2.transpose()

def ntk_matrix(X1,X2,model,optimizer):
    # Xi must be a torch variable
    Kappa = np.empty((len(X1),len(X2)))
    with tqdm(total=int(len(X1)*len(X2))) as pbar:
        for i1,x1 in enumerate(X1):
            if type(x1) is tuple:
                x1,_ = x1
            for i2,x2 in enumerate(X2):
                if type(x2) is tuple:
                    x2,_ = x2
                Kappa[i1,i2] = ntk_single(x1,x2,model,optimizer)
                pbar.update(1)
    return Kappa

def MVP_JTX(v,model,X_training, optimizer):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mvp = np.zeros((p,1))
    for i,(xi,_) in enumerate(X_training):
        xi = torch.from_numpy(xi)
        g = gradient_model(model=model,optimizer=optimizer,xi=xi).reshape((p,1))
        mvp += v[i]*g
    return mvp

def MVP_JX(v,model,X_training, optimizer):
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = len(X_training)
    mvp = np.zeros((n,1))
    v = v.reshape((p,1))
    for i,(xi,_) in enumerate(X_training):
        xi = torch.from_numpy(xi)
        g = gradient_model(model=model,optimizer=optimizer,xi=xi).reshape((p,1))
        mvp[i,0] = g.transpose() @ v
    return mvp

def MVP_JJT(v,model,X_training, optimizer):
    x1 = MVP_JTX(v,model,X_training, optimizer)
    x2 = MVP_JX(x1,model,X_training, optimizer)
    return x2

def ensemble_result(test_loader, ensemble_M, model_list, sy=1, my=0):
    mu_test_list = np.empty((ensemble_M,len(test_loader.dataset)))
    sigma_test_list = np.empty((ensemble_M,len(test_loader.dataset)))
    for i in range(ensemble_M):
        for X,y in test_loader:
            y = y.reshape((y.shape[0],1))
            mu, sig = model_list[i](X)
            mu = mu * sy + my
            sig = sig * (sy**2)
            mu_test_list[i,:] = np.reshape(to_np(mu), (len(test_loader.dataset)))
            sigma_test_list[i,:] = np.reshape(to_np(sig),(len(test_loader.dataset)))
    mu_mean = np.mean(mu_test_list,axis=0)
    sigma_mean = np.mean(sigma_test_list, axis=0) + np.mean(np.square(mu_test_list), axis = 0) - np.square(mu_mean)
    return mu_mean, sigma_mean

def calibration_curve_ntk(testloader, uncertainties, model, num_c,my=0,sy=1):
    c = np.linspace(0,1,num_c)
    observed_true = np.empty(num_c)
    total = uncertainties.size
    for i, (X,y) in enumerate(testloader):
         Yhat_pre = model(X)
         Yhat = Yhat_pre.detach().numpy()*sy + my
         y = y.detach().numpy()
    for i,ci in enumerate(c):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_c = z * np.sqrt(uncertainties*(sy**2))
        left_ci = y >= (Yhat - ci_c.reshape(-1,1)).squeeze(1)
        right_ci = y <= (Yhat + ci_c.reshape(-1,1)).squeeze(1)
        observed_true_c = np.logical_and(left_ci,right_ci)
        num_true = observed_true_c[observed_true_c==True].size
        observed_true[i] = num_true/total
        # print(num_true)
    return observed_true

def calibration_curve_ensemble(testloader, mu, sigma2, num_c):
    c = np.linspace(0,1,num_c)
    observed_true = np.empty(num_c)
    total = mu.size
    for i, (_,y) in enumerate(testloader):
         y = y.detach().numpy()
    for i,ci in enumerate(c):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_c = z * np.sqrt(sigma2)
        left_ci = y >= (mu - ci_c)
        right_ci = y <= (mu + ci_c)
        observed_true_c = np.logical_and(left_ci,right_ci)
        num_true = observed_true_c[observed_true_c==True].size
        observed_true[i] = num_true/total
        # print(num_true)
    return observed_true

def plot_calibration(observed_true_ntk, observed_true_ensemble, dataset_str, dir_name, plot_name):
    num_c = observed_true_ntk.size
    c = c = np.linspace(0,1,num_c)
    plt.plot(c,c)
    plt.plot(c,observed_true_ntk, label='NTK')
    plt.plot(c,observed_true_ensemble, label='Deep Ensemble')
    plt.xlabel("Expected accuracy")
    plt.ylabel("Observed accuracy")
    plt.title("Calibration curve".format(dataset_str))
    plt.legend()
    plt.savefig(dir_name + plot_name, format="pdf", bbox_inches="tight")
    plt.show()

### Classification definitions
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 3, 5)
        self.fc1 = nn.Linear(3 * 4 * 4, 10)
        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(84,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x,len(x.shape)-3)
        x = self.fc1(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x    

def train_loop(dataloader, model, loss_fn, optimizer, train_mode=True):
    size = len(dataloader.dataset)
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        total_loss += loss
    total_loss_mean = total_loss / (batch + 1)
    return total_loss_mean


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss, correct
        
def general_probit_approx(mu,sigma):
    return mu/(np.sqrt(1+np.pi/8*sigma))




def ntk_method(train, test, model,  filepath, num_class: int = 10, solver: str = 'direct_direct', 
               batch_size_eval: int = 0, batch_size_K: int = 0, batch_size_kxX : int = 0, reg = 0, 
               softmax: bool = False, cr_maxit: int = 100, cr_rtol: float = 1e-12, lsmr_maxit: int = 30, verbose = False):
    '''
    Calculates mu,sigma2 for all points in test, for model trained on train set.

    solver options:
     - 'direct_direct'
     - 'direct_iterative'
     - 'iterative_iterative_cr'
     - 'iterative_iterative_lsmr'
    '''
    print("Start method")
    ("Memory used = {}GB \n".format(torch.cuda.memory_allocated(device)*1e-9))
    
    ### Model info/fnet
    fnet, params, buffers = make_functional_with_buffers(model)
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ### Set up storage
    sigma2 = torch.empty((len(test),num_class),device='cpu')
    mu = torch.empty((len(test),num_class),device='cpu')

    ### Datasets and DataLoaders
    test_loader = DataLoader(test,1)
    train_loader = DataLoader(train,len(train))
    train_loader_individual = DataLoader(train,1)
    train_x,train_y = next(iter(train_loader))
    train_x,train_y = train_x.to(device), train_y.to(device)
    if batch_size_eval > 0:
        train_batch_size = batch_size_eval
    elif batch_size_eval == 0:
        train_batch_size = len(train)
    train_loader_batched = DataLoader(train,train_batch_size)
    
    ### Residual (y-f) for mu
    print("Finding Residual for NTK Method")
    print("Memory used = {}GB \n".format(torch.cuda.memory_allocated(device)*1e-9))
    torch.cuda.empty_cache()
    with torch.no_grad():
        train_y = one_hot(train_y,num_classes=num_class)
        if softmax:
            train_y_hat = model(train_x).softmax(dim=1)
        else:
            print("Evaluating model for NTK Method")
            print("Memory used = {}GB \n".format(torch.cuda.memory_allocated(device)*1e-9))
            train_y_hat = []
            for i,(x,_) in enumerate(train_loader_batched):
                train_y_hat.append(model(x.to(device)))
            train_y_hat = torch.cat(train_y_hat,dim=0)
        train_residual = train_y - train_y_hat

    ### Make sure ntk folder exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with tqdm(total=int(len(test)*num_class)) as pbar:
        ### Directly form Kappa
        if solver == 'direct_direct' or solver == 'direct_iterative':
            for c in range(num_class):

                ### Take class c from output of model
                if not softmax: # For model not trained on brier loss
                    def fnet_single(params, x):
                        return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)[c].reshape(1)
                    
                else: # For model trained on brier loss
                    def fnet_single(params, x):
                        return fnet(params, buffers, x.unsqueeze(0)).squeeze(0).softmax(dim=0)[c].reshape(1)
                
                ### Form kappa
                print("Memory used = {}GB \n".format(torch.cuda.memory_allocated(device)*1e-9))

                ntkpath = filepath + '/ntk_{}.npy'.format(c)
                if os.path.exists(ntkpath):
                    print("Accessing Kappa {} from memory".format(c))
                    Kappa = np.memmap(filename=ntkpath,mode='c',dtype='float64',shape=((len(train),len(train))))
                    Kappa = torch.from_numpy(Kappa)
                else:
                    print("Creating Kappa {} and storing in memory".format(c))
                    Kappa = empirical_ntk_memmap(fnet_single,params,train,ntkpath,batch_size_K)
                    Kappa = torch.from_numpy(Kappa)

                ## Add regularising constant if matrix ill-conditioned
                print("reg type :: {}".format(reg))
                Kappa += reg*torch.eye(len(train))

                cholpath = filepath + '/chol_{}.npy'.format(c)
                if os.path.exists(cholpath):
                    print("Accessing Cholesky {} from memory".format(c))
                    cholesky_K = np.memmap(filename=cholpath,mode='c',dtype='float64',shape=((len(train),len(train))))
                    cholesky_K = torch.from_numpy(cholesky_K).cpu()
                else:
                    print("Creating Cholesky {} and storing in memory".format(c))
                    chol_c = np.memmap(filename=cholpath,mode='w+',dtype='float64',shape=((len(train),len(train))))
                    cholesky_K = torch.linalg.cholesky(Kappa.to(device)).cpu()
                    chol_c[:,:] = cholesky_K
                    chol_c.flush()

                print("Solving for test points")
                for i,(x_t,_) in enumerate(test_loader):
                    x_t = x_t.to(device)

                    ### Form small kappas
                    kappa_xX = empirical_kappa_xX(fnet_single,params,train,x_t,batch_size_kxX).detach()
                    kappa_xx = empirical_kappa_xX(fnet_single,params,x_t,x_t,batch_size_kxX).detach()

                    ### Directly solve K^-1
                    if solver == 'direct_direct':
                        y = torch.linalg.solve_triangular(cholesky_K.to(device),kappa_xX.to(device),upper=False)
                        Kappa_solve = torch.linalg.solve_triangular(cholesky_K.T.to(device),y.to(device),upper=True)

                    ### Iteratively solve K^-1 kappa_xX
                    elif solver == 'direct_iterative':
                        Kappa_solve,_,_,_,_ = solv.CR_torch(Kappa.to(device),kappa_xX.reshape(-1),rtol=1e-12,maxit=100,VERBOSE=False)
                        Kappa_solve = Kappa_solve.to("cpu")

                    ### Uncertainty value
                    sigma2_i = kappa_xx.to(device) - kappa_xX.reshape(1,-1).to(device) @ Kappa_solve.reshape(-1,1).to(device)
                    sigma2[i,c] = sigma2_i

                    ### Mean value
                    mu_i = Kappa_solve.reshape(1,-1).to(device) @ train_residual[:,c].reshape(-1,1).to(device) + model(x_t).squeeze(0)[c].to(device)
                    mu[i,c] = mu_i

                    ### Update progress bar
                    pbar.update(1)

                del Kappa
                del cholesky_K
                torch.cuda.empty_cache()

        
        ### Don't directly form Kappa - use MVP
        if solver == 'iterative_iterative_cr' or solver == 'iterative_iterative_lsmr':
            for c in range(num_class):
                ### Take class c from output of model
                def fnet_single(params, x):
                    return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)[c].reshape(1)
                
                ### Define MVPs for fnet(c)
                Ax = lambda x: JTw(x,fnet_single,params,train_loader_individual)
                ATx = lambda x: Jw(x,fnet_single,params,train_loader_individual)
                Kappa = lambda x: ATx(Ax(x))

                if solver == 'iterative_iterative_lsmr':
                    ### LinearOperator for use with scipy.linalg.lsmr
                    A = LinearOperator((num_weights,len(train)),matvec=Ax,rmatvec=ATx)

                for i,(x_t,_) in enumerate(test_loader):
                    x_t = x_t.to(device)

                    ### Form small kappas
                    kappa_xx = empirical_kappa_xX(fnet_single,params,x_t,x_t,batch_size_kxX).detach()
                    kappa_xX = ATx(grad_f(x_t,fnet_single,params)).detach() # we form this way as using empirical_ntk takes 8 * n * p bytes.

                    ### Iteratively solve K^-1 kappa_xX using MVP and lsmr
                    if solver == 'iterative_iterative_lsmr':
                        ### b = grad_f(test_point)
                        b = grad_f(x_t,fnet_single,params).cpu().detach().numpy() #lsmr is a numpy function, must cast to numpy
                        Kappa_solve = lsmr(A,b,show=False,maxiter=lsmr_maxit)
                        Kappa_solve = torch.from_numpy(Kappa_solve[0]).detach()

                    ### Iteratively solve K^-1 kappa_xX using MVP and CR
                    elif solver == 'iterative_iterative_cr':
                        print("Solving CR")
                        Kappa_solve = solv.CR_torch(Kappa,kappa_xX.reshape(-1).to(device),rtol=cr_rtol,maxit=cr_maxit,VERBOSE=True)
                        Kappa_solve = Kappa_solve[0].detach().cpu()

                    ### Uncertainty value
                    sigma2_i = kappa_xx.to(device) - kappa_xX.reshape(1,-1).to(device) @ Kappa_solve.reshape(-1,1).to(device)
                    sigma2[i,c] = sigma2_i

                    ### Mean value
                    mu_i = Kappa_solve.reshape(1,-1).to(device) @ train_residual[:,c].reshape(-1,1) + model(x_t).squeeze(0)[c]
                    mu[i,c] = mu_i

                    ### Update progress bar
                    pbar.update(1)

    return mu, sigma2

def one_hot(y,num_classes):
        yh = torch.zeros((y.shape[0],num_classes),dtype=torch.float64,device=device)
        yh[torch.arange(y.shape[0]),y] = 1
        return yh

def empirical_kappa_xX(fnet_single, params, x1, x2, batch_size=0, verbose = False):

    '''
        Only use for non symmetrical ntk_xX

        INPUT:
            - fnet_single: must be function form of single-output NN
            - params:
            - x1: either a torch.dataset/torch.subset type or a list containing a single torch.tensor
            - x2: either a torch.dataset/torch.subset type or a list containing a single torch.tensor
            - batch_size: (int) size of dataset to calculate eNTK in parallel. Larger values take more storage, lower values take longer to calculate. 
                Batch size of 0 equates to elementwise calculation.

        OUTPUT:
            - ntk: torch.tensor of size len(x1) x len(x2).
        '''
    # Set the seed for random transforms on dataset
    seed = 2849571098

    # Create fixed dataset and loader
    dataset_x1 = [x for _,x in enumerate(x1)]
    dataset_x2 = [x for _,x in enumerate(x2)]
    if batch_size > 0:
        block_size = batch_size
    elif batch_size == 0:
        block_size = 1
    x1_loader = DataLoader(dataset_x1,block_size,shuffle=False)
    x2_loader = DataLoader(dataset_x2,block_size,shuffle=False)

    # Setup loop variables
    if verbose:
        pbar = tqdm(total=int(len(x1_loader)*(len(x2_loader))))
    nidx,nidy = 0,0
    ntk = torch.empty((len(dataset_x1),len(dataset_x2)),dtype=torch.float64)
    
    # # Set random transforms to deterministic (need to do it to both loops)
    # torch.random.manual_seed(1)

    for _,x1_batch in enumerate(x1_loader):
        # Setup data
        if len(x1_batch)>1:
            x1_batch,_ = x1_batch
        x1_batch = x1_batch.to(device=device,dtype=torch.float64)

        # # Set random transforms to deterministic (need to do it to both loops)
        # torch.random.manual_seed(1)

        for _,x2_batch in enumerate(x2_loader):
            # Setup data
            if len(x2_batch)>1:
                x2_batch, _ = x2_batch

            x2_batch = x2_batch.to(device=device,dtype=torch.float64)

            # Compute block element of ntk
            J, _ = empirical_ntk_jacobian_contraction(fnet_single, params, x1_batch, x2_batch)
            J = J.cpu()

            # Place block element in ntk
            incx = J.shape[0]
            incy = J.shape[1]
            ntk[nidx:nidx+incx,nidy:nidy+incy] = J

            # Move to next batch of coloumns and update progress bar    
            nidy += incy
            if verbose:
                pbar.update(1)

        # Move to next batch of rows, start at coloumn zero
        nidx += incx
        nidy = 0
    return ntk

def empirical_ntk_local(fnet_single, params, training_data, filename, batch_size=0, verbose = False):
    '''
    Only use for symmetrical full NTK_XX

    INPUT:
        - fnet_single: must be function form of single-output NN
        - params:
        - training_data: either a torch.dataset/torch.subset type or a list containing a single torch.tensor
        - filename: file path to store ntk matrix
        - batch_size: (int) size of dataset to calculate eNTK in parallel. Larger values take more storage, lower values take longer to calculate. 
            Batch size of 0 equates to elementwise calculation.

    OUTPUT:
        - ntk: torch.tensor of size len(training_data) x len(training_data).
    '''
    # Set the seed for random transforms on dataset
    seed = 2849571098

    # Create fixed dataset and loader
    dataset_fixed = [x for _,x in enumerate(training_data)]
    if batch_size > 0:
        block_size = batch_size
    elif batch_size == 0:
        block_size = 1
    train_loader = DataLoader(dataset_fixed,block_size,shuffle=False)

    # Setup loop variables
    if verbose:
        pbar = tqdm(total=int(len(train_loader)*(len(train_loader)+1)/2))
    nidx,nidy = 0,0
    ntk = torch.empty((len(training_data),len(training_data)),dtype=torch.float64)
    
    # Progress bar

    # # Set random transforms to deterministic (need to do it to both loops)
    # torch.random.manual_seed(1)

    for idx,x1_batch in enumerate(train_loader):
        # Setup data
        if len(x1_batch)>1:
            x1_batch,_ = x1_batch
        x1_batch = x1_batch.to(device=device,dtype=torch.float64)

        # Refresh jacobian for each row, so it is calculated fresh
        jac1 = None

        # # Set random transforms to deterministic (need to do it to both loops)
        # torch.random.manual_seed(1)

        for idy,x2_batch in enumerate(train_loader):
            # Setup data
            if len(x2_batch)>1:
                x2_batch, _ = x2_batch

            # Only compute upper triangular elements (due to symmetry)
            if idy < idx:
                nidy += x2_batch.shape[0]
                continue

            x2_batch = x2_batch.to(device=device,dtype=torch.float64)

            # Compute block element of ntk
            J, jac1 = empirical_ntk_jacobian_contraction(fnet_single, params, x1_batch, x2_batch, jac1 = jac1)
            J = J.cpu()

            # Place block element in ntk
            incx = J.shape[0]
            incy = J.shape[1]
            ntk[nidx:nidx+incx,nidy:nidy+incy] = J

            # Matrix is symmetrical
            if idx != idy:
                ntk[nidy:nidy+incy,nidx:nidx+incx] = J.T

            # Move to next batch of coloumns and update progress bar    
            nidy += incy
            if verbose:
                pbar.update(1)

        # Move to next batch of rows, start at coloumn zero
        nidx += incx
        nidy = 0
    return ntk

def empirical_ntk_memmap(fnet_single, params, training_data, filename, batch_size=0):
    '''
    Only use for symmetrical full NTK_XX

    INPUT:
        - fnet_single: must be function form of single-output NN
        - params:
        - training_data: either a torch.dataset/torch.subset type or a list containing a single torch.tensor
        - filename: file path to store ntk matrix
        - batch_size: (int) size of dataset to calculate eNTK in parallel. Larger values take more storage, lower values take longer to calculate. 
            Batch size of 0 equates to elementwise calculation.

    OUTPUT:
        - ntk: torch.tensor of size len(training_data) x len(training_data).
    '''
    # Set the seed for random transforms on dataset
    seed = 2849571098

    # Create fixed dataset and loader
    dataset_fixed = [x for _,x in enumerate(training_data)]
    if batch_size > 0:
        block_size = batch_size
    elif batch_size == 0:
        block_size = 1
    train_loader = DataLoader(dataset_fixed,block_size,shuffle=False)

    # Setup loop variables
    nidx,nidy = 0,0
    ntk = np.memmap(filename=filename,dtype='float64', mode = 'w+', shape=(len(training_data),len(training_data)))
    
    # Progress bar
    with tqdm(total=int(len(train_loader)*(len(train_loader)+1)/2)) as pbar:

        # # Set random transforms to deterministic (need to do it to both loops)
        # torch.random.manual_seed(1)

        for idx,x1_batch in enumerate(train_loader):
            # Setup data
            if len(x1_batch)>1:
                x1_batch,_ = x1_batch
            x1_batch = x1_batch.to(device=device,dtype=torch.float64)

            # Refresh jacobian for each row, so it is calculated fresh
            jac1 = None

            # # Set random transforms to deterministic (need to do it to both loops)
            # torch.random.manual_seed(1)

            for idy,x2_batch in enumerate(train_loader):
                # Setup data
                if len(x2_batch)>1:
                    x2_batch, _ = x2_batch

                # Only compute upper triangular elements (due to symmetry)
                if idy < idx:
                    nidy += x2_batch.shape[0]
                    continue
   
                x2_batch = x2_batch.to(device=device,dtype=torch.float64)

                # Compute block element of ntk
                J, jac1 = empirical_ntk_jacobian_contraction(fnet_single, params, x1_batch, x2_batch, jac1 = jac1)
                J = J.cpu()

                # Place block element in ntk
                incx = J.shape[0]
                incy = J.shape[1]
                ntk[nidx:nidx+incx,nidy:nidy+incy] = J

                # Matrix is symmetrical
                if idx != idy:
                    ntk[nidy:nidy+incy,nidx:nidx+incx] = J.T

                # Move to next batch of coloumns and update progress bar    
                nidy += incy
                pbar.update(1)

            # Move to next batch of rows, start at coloumn zero
            nidx += incx
            nidy = 0
    return ntk

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, jac1 = None):
        
        if jac1 is None:
            # Compute J(x1)
            jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
            jac1 = [j.detach().flatten(2).flatten(0,1) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.detach().flatten(2).flatten(0,1) for j in jac2]

        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)

        return result, jac1

def liam_ntk(fnet_single, params, training_data, filename):
    trainloader = DataLoader(training_data)
    dataset = [(idx, x) for idx, (x, _) in enumerate(trainloader)]
    # Get number of data points
    N = 0
    N = sum([x.shape[0] for (idx, x) in dataset])
    ntk = np.memmap(filename, dtype='float64', mode='w+', shape=(N,N))
    nidx = 0
    nidy = 0
    with tqdm(total=int(len(dataset)*(len(dataset)+1)/2)) as pbar:
        for idx, x1 in dataset:
            for idy, x2 in dataset:
                if idy < idx:
                    nidy += x2.shape[0]
                    continue
                J = liam_empirical_ntk(fnet_single, params, x1.to(device), x2.to(device)).cpu()
                incx = J.shape[0]
                incy = J.shape[1]
                ntk[nidx:nidx+incx,nidy:nidy+incy] = J
                if idx != idy:
                    ntk[nidy:nidy+incy,nidx:nidx+incx] = J.T
                nidy += incy
                pbar.update(1)
            nidx += incx
            nidy = 0
    return ntk

def liam_empirical_ntk(fnet_single, params, x1, x2, jac1):
        
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j.detach().flatten(2).flatten(0,1) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.detach().flatten(2).flatten(0,1) for j in jac2]

        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Nf,Mf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result

def ntk_uncertainty_explicit_class(train_dataset, test_dataset, model, num_classes, type='direct', rtol=1e-9, maxit=50,softmax=False):
    fnet, params = make_functional(model)    
    uncertainty_array = np.empty((num_classes,len(test_dataset)))
    mu = np.empty((num_classes,len(test_dataset)))
    test_dataloader = DataLoader(test_dataset)
    train_dataloader = DataLoader(train_dataset,len(train_dataset))
    X_train,y_train = next(iter(train_dataloader))

    ### One hot for brier loss
    def one_hot(y,num_classes):
        yh = torch.zeros((y.shape[0],num_classes),dtype=torch.float64)
        yh[torch.arange(y.shape[0]),y] = 1
        return yh

    if softmax:
        f_train = model(X_train).softmax(dim=1)
        y_train = one_hot(y_train,num_classes=num_classes)
    else:
        f_train = model(X_train)
        y_train = one_hot(y_train,num_classes=num_classes)
    resid_hat = (y_train - f_train).detach().numpy()

    with tqdm(total=int(len(test_dataset)*num_classes)) as pbar:
        for c in range(num_classes):
            def fnet_single(params, x):
                if softmax:
                    f = fnet(params, x.unsqueeze(0)).squeeze(0).softmax(dim=0)[c].reshape(1) 
                else:
                    f = fnet(params, x.unsqueeze(0)).squeeze(0)[c].reshape(1) 
                return f
            
            Kappa = empirical_ntk_jacobian_contraction(
                fnet_single=fnet_single,
                params=params,
                x1=X_train,
                x2=X_train
                ).detach().numpy().squeeze((2,3))
    
            for i,(x_test,_) in enumerate(test_dataloader):
                kappa_xx = empirical_ntk_jacobian_contraction(
                    fnet_single=fnet_single,
                    params=params,
                    x1=x_test,
                    x2=x_test
                    ).detach().numpy().squeeze((2,3))
                kappa_xX = empirical_ntk_jacobian_contraction(
                    fnet_single=fnet_single,
                    params=params,
                    x1=x_test,
                    x2=X_train
                    ).detach().numpy().squeeze((2,3))
                if type=='direct':
                    uncertainty_estimate = kappa_xx - kappa_xX @ np.linalg.solve(Kappa,kappa_xX.transpose())
                elif type=='iterative':
                    x_solve, it, resid, rel_resid, rel_mat_resid = solv.CR(
                        Kappa,
                        kappa_xX.transpose(),
                        rtol=rtol,
                        init=False, 
                        maxit=maxit, 
                        VERBOSE=False
                        )
                    
                    # kappa_hat = lifted_solution(x_solve,resid)
                    uncertainty_estimate = kappa_xx - kappa_xX @ x_solve
                    # lifted_ue = kappa_xx - kappa_hat.transpose() @ Kappa @ kappa_hat
                    # uncertainty_array_lift[0,i] = lifted_ue
                    fx = model(x_test)
                    mu_x = x_solve.transpose() @ resid_hat[:,c].reshape((-1,1)) + fx.detach().numpy().squeeze(0)[c]
                uncertainty_array[c,i] = uncertainty_estimate
                mu[c,i] = mu_x
                pbar.update(1)
    
    return uncertainty_array, mu

def lifted_solution(x,r):
    x = x - ((r.transpose() @ x) / np.linalg.norm(r)) * r
    return x

def lifted_solution_torch(x,r):
    x = x - ((r.transpose(0,1) @ x) / torch.norm(r)) * r
    return x

def tensor_to_block_mat(mat):
    '''
    Mat must have dimension n1 x n2 x n3 x n4, will convert
    to 2d matrix M where at M(i,j) is a n3 x n4 block matrix from tensor.
    '''
    s = mat.shape
    mat = mat.transpose(1,2).reshape(s[0]*s[2],s[1]*s[3])
    return mat

def calibration_curve_values(preds, target, n_bins):
            '''
            INPUTS:
                - preds :: (n x c) torch.tensor
                - targets :: (n) torch.tensor

            OUTPUTS:
                - xs :: (n_bins) torch.tensor
                    average confidence of prediction in each bin, given by softmax probability
                - ys :: (n_bins) torch.tensor
                    average accuracy of predictions in each bin
            '''

            confidences, predictions = preds.softmax(dim=1).max(1)
            step = (confidences.shape[0] + n_bins -1) // n_bins
            bins = torch.sort(confidences)[0][::step]
            if confidences.shape[0] % step != 1:
                bins = torch.cat((bins, confidences.max().reshape(1)),dim=0)
            bin_lowers = bins[:-1]
            bin_uppers = bins[1:]
            accuracies = predictions == target
            xs = []
            ys = []
            for bin_lower_conf,bin_upper_conf in zip(bin_lowers,bin_uppers):
                in_bin = (confidences > bin_lower_conf) * (confidences < bin_upper_conf)
                prop_in_bin = in_bin.double().mean(dtype=torch.double)
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].double().mean(dtype=torch.double)
                    confidence_in_bin = confidences[in_bin].double().mean(dtype=torch.double)
                    xs.append(confidence_in_bin)
                    ys.append(accuracy_in_bin)
            xs = torch.tensor(xs)
            ys = torch.tensor(ys)

            return xs, ys

def evaluate_batch(dataset,model,batch_size):
            if batch_size > 0:
                bs = batch_size
            elif batch_size == 0:
                bs = len(dataset)
            loader_batched = DataLoader(dataset,bs)
            evaluate = []
            for b,(x,_) in enumerate(loader_batched):
                evaluate.append(model(x.to(device)).detach())
            evaluate = torch.cat(evaluate,dim=0)
            return evaluate

def evaluate_batch_T(dataset,model,batch_size):
            if batch_size > 0:
                bs = batch_size
            elif batch_size == 0:
                bs = len(dataset)
            loader_batched = DataLoader(dataset,bs)
            evaluate = []
            for b,(x,_) in enumerate(loader_batched):
                evaluate.append(model(x.to(device)).detach()) # output is (samples x batchsize x output)
            evaluate = torch.cat(evaluate,dim=1)
            return evaluate