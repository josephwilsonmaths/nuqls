import torch
from functorch import make_functional
from torch.func import vmap, jacrev
from torch.utils.data import DataLoader
import tqdm
import copy
import time
from torch.profiler import profile, record_function, ProfilerActivity
import tracemalloc

import posteriors.nuqlsPosterior.nuqlsUtils as nqlutil

torch.set_default_dtype(torch.float64)

class regressionParallelFull(object):
    '''
    NUQLS implementation for: regression, larger S, tiny dataset size. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,network):
        self.network = network
        self.device = next(network.parameters()).device

    def train(self, train, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False, progress_bar=True):
        fnet, params = make_functional(self.network)
        theta_t = nqlutil.flatten(params)
        theta = torch.randn(size=(theta_t.shape[0],S),device=self.device)*scale + theta_t.unsqueeze(1)

        ## Create loaders and get entire training set
        train_loader_total = DataLoader(train,batch_size=len(train))
        X,Y = next(iter(train_loader_total))

        ## Compute jacobian of net, evaluated on training set
        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        J = vmap(jacrev(fnet_single), (None, 0))(params, X.to(self.device))
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()

        # Set progress bar
        if progress_bar:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        ## Train S realisations of linearised networks
        for epoch in pbar:
            if self.device == torch.device('cuda'):
                torch.cuda.reset_peak_memory_stats()

            X, Y = X.to(self.device), Y.to(self.device).reshape(-1,1)
            
            f_nlin = self.network(X)
            f_lin = (J.to(self.device) @ (theta - theta_t.unsqueeze(1)) + f_nlin).detach()
            resid = f_lin - Y
            grad = J.T.to(self.device) @ resid.to(self.device) / X.shape[0]

            if epoch == 0:
                bt = grad
            else:
                bt = mu*bt + grad

            theta -= lr * bt

            loss = torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).max()

            if verbose:
                metrics = {'mean_loss': loss.item(),
                   'resid_norm': torch.mean(torch.square(grad)).item()}
                if self.device == torch.device('cuda'):
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

        # Report maximum loss over S, and the mean gradient norm
        max_l2_loss = torch.mean(torch.square(J @ (theta - theta_t.unsqueeze(1)) + f_nlin - Y)).item()
        norm_resid = torch.mean(torch.square(J.T @ ( f_nlin + J @ (theta - theta_t.unsqueeze(1)) - Y))).item()

        if verbose:
            print('Posterior samples computed!')
        self.theta = theta

        return max_l2_loss, norm_resid
    
    def test(self, test, test_bs=1000):
        fnet, params = make_functional(self.network)
        theta_t = nqlutil.flatten(params)

        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        test_loader = DataLoader(test,batch_size=test_bs)
                
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            J = vmap(jacrev(fnet_single), (None, 0))(params, x)
            J = [j.detach().flatten(1) for j in J]
            J = torch.cat(J,dim=1)

            pred_lin = J @ (self.theta - theta_t.unsqueeze(1)) + f_nlin ## n x S
            pred_s.append(pred_lin.detach())

        predictions = torch.cat(pred_s,dim=0)
        return predictions