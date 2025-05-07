import torch
from functorch import make_functional
from torch.func import vmap, jacrev
from torch.utils.data import DataLoader
import tqdm
import copy
import time
from torch.profiler import profile, record_function, ProfilerActivity
import tracemalloc

from posteriors.nuqlsPosterior.nuqlsUtils import *

torch.set_default_dtype(torch.float64)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class regressionParallelFull(object):
    '''
    NUQLS implementation for: regression, larger S, tiny dataset size. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,network):
        self.network = network

    def train(self, train, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False, progress_bar=True):
        fnet, params = make_functional(self.network)
        theta_t = flatten(params)
        theta = torch.randn(size=(theta_t.shape[0],S),device=device)*scale + theta_t.unsqueeze(1)
        
        ## Create new parameter to train
        self.theta_init = self.theta.clone().detach()

        ## Create loaders and get entire training set
        train_loader_total = DataLoader(train,batch_size=len(train))
        X,Y = next(iter(train_loader_total))

        ## Compute jacobian of net, evaluated on training set
        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        J = vmap(jacrev(fnet_single), (None, 0))(params, X.to(device))
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()

        # Set progress bar
        if progress_bar:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        ## Train S realisations of linearised networks
        for epoch in pbar:
            X, Y = X.to(device), Y.to(device).reshape(-1,1)
            
            f_nlin = self.network(X)
            f_lin = (J.to(device) @ (theta - theta_t.unsqueeze(1)) + f_nlin).detach()
            resid = f_lin - Y
            grad = J.T.to(device) @ resid.to(device) / X.shape[0]

            if epoch == 0:
                bt = grad
            else:
                bt = mu*bt + grad

            theta -= lr * bt

            loss = torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).max()

            if epoch % 100 == 0 and verbose:
                print("\n-----------------")
                print("Epoch {} of {}".format(epoch,epochs))
                print("max l2 loss = {}".format(torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).max()))
                print("Residual of normal equation l2 = {}".format(torch.mean(torch.square(J.T @ ( f_nlin + J @ (theta - theta_t.unsqueeze(1)) - Y)))))

        # Report maximum loss over S, and the mean gradient norm
        max_l2_loss = torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).max()
        norm_resid = torch.mean(torch.square(J.T @ ( f_nlin + J @ (theta - theta_t.unsqueeze(1)) - Y)))

        if verbose:
            print('Posterior samples computed!')
        self.theta = theta

        return max_l2_loss.detach(), norm_resid.detach()
    
    def test_linear(self,test,my=0,sy=1):
        fnet, params = make_functional(self.network)
        theta_t = flatten(params)

        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        test_loader = DataLoader(test,batch_size=len(test))
                
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            f_nlin = self.network(x)
            J = vmap(jacrev(fnet_single), (None, 0))(params, x)
            J = [j.detach().flatten(1) for j in J]
            J = torch.cat(J,dim=1)

            pred_lin = J @ (self.theta - theta_t.unsqueeze(1)) + f_nlin ## n x S
            pred_s.append(pred_lin.detach()*sy + my)

        predictions = torch.cat(pred_s,dim=1)
        return predictions