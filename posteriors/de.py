import torch
import tqdm
import utils.training
import utils.optimizers
import copy

def DeepEnsemble(
    network: torch.nn.Module,
    task: str = 'regression',
    M: int = 5
    ):

    if task == 'regression':
        return DeepEnsembleRegression(network, M)
    
    elif task == 'classification':
        return DeepEnsembleClassification(network, M)
    
    else:
        print("Invalid task choice. Valid choices: [regression, classification]")


class DeepEnsembleRegression(object):
    def __init__(self, network, M = 5):
        self.network = network
        self.device = next(network.parameters()).device
        self.M = M
        self.network_list = [copy.deepcopy(self.network) for _ in range(self.M)]
        for net in self.network_list:
            net.apply(utils.training.init_weights)
        
    def train(self, loader, lr, wd, epochs, optim_name, sched_name, verbose=False, extra_verbose=False):
        opt_list = []
        sched_list = []

        for i in range(self.M):
            opt, sched = utils.optimizers.get_optim_sched(self.network_list[i], optim_name, sched_name, lr, wd, epochs)
            opt_list.append(opt)
            sched_list.append(sched)


        total_nll, total_mse = 0, 0

        if verbose:
            pbar = tqdm.trange(self.M)
        else:
            pbar = range(self.M)

        nll_func = torch.nn.GaussianNLLLoss()
        mse_func = torch.nn.MSELoss(reduction='mean')

        for idx in pbar:

            if extra_verbose:
                pbar_inner = tqdm.trange(epochs)
            else:
                pbar_inner = range(epochs)

            for epoch in pbar_inner:

                train_nll, train_mse = train_loop(loader, self.network_list[idx], opt_list[idx], nll_func, mse_func, scheduler=sched_list[idx], device=self.device)
                
                if extra_verbose:
                    metrics = {'train nll': train_nll,
                    'train mse': train_mse}
                    pbar_inner.set_postfix(metrics)

            total_nll += train_nll
            total_mse += train_mse

            if verbose:
                metrics = {'train nll': train_nll,
                   'train mse': train_mse}
                pbar.set_postfix(metrics)

        total_nll /= self.M
        total_mse /= self.M

        return total_nll, total_mse
    
    def test(self, loader):
        ensemble_pred = []
        ensemble_var = []
        for idx in range(self.M):
            network_pred = []
            network_var = []
            for x,_ in loader:
                pred, var = self.network_list[idx](x.to(self.device))
                network_pred.append(pred)
                network_var.append(var)
            ensemble_pred.append(torch.cat(network_pred))
            ensemble_var.append(torch.cat(network_var))
        ensemble_pred = torch.stack(ensemble_pred)
        ensemble_var = torch.stack(ensemble_var)

        mean_pred = torch.mean(ensemble_pred,dim=0)
        var_pred = torch.mean(ensemble_var + torch.square(ensemble_pred), dim=0) - torch.square(mean_pred)

        return mean_pred.detach(), var_pred.detach()
        
class DeepEnsembleClassification(object):
    def __init__(self, network, M = 5):
        self.network = network
        self.device = next(network.parameters()).device
        self.M = M
        self.network_list = [copy.deepcopy(self.network) for _ in range(self.M)]
        for net in self.network_list:
            net.apply(utils.training.init_weights)

    def train(self, loader, lr, wd, epochs, optim_name, sched_name, verbose=False, extra_verbose=False):
        opt_list = []
        sched_list = []

        for i in range(self.M):
            opt, sched = utils.optimizers.get_optim_sched(self.network_list[i], optim_name, sched_name, lr, wd, epochs)
            opt_list.append(opt)
            sched_list.append(sched)


        total_nll, total_acc = 0, 0

        if verbose:
            pbar = tqdm.trange(self.M)
        else:
            pbar = range(self.M)

        nll_func = torch.nn.CrossEntropyLoss()

        for idx in pbar:

            if extra_verbose:
                pbar_inner = tqdm.trange(epochs)
            else:
                pbar_inner = range(epochs)

            for epoch in pbar_inner:

                train_nll, train_acc = utils.training.train_loop(dataloader=loader, 
                                                                 model=self.network_list[idx],
                                                                 loss_fn=nll_func,
                                                                 optimizer=opt_list[idx],
                                                                 scheduler=sched_list[idx],
                                                                 device=self.device)
                
                if extra_verbose:
                    metrics = {'train nll': train_nll,
                    'train acc': train_acc}
                    pbar_inner.set_postfix(metrics)

            total_nll += train_nll
            total_acc += train_acc

            if verbose:
                metrics = {'train nll': train_nll,
                   'train mse': train_acc}
                pbar.set_postfix(metrics)

        total_nll /= self.M
        train_acc /= self.M

        return total_nll, train_acc
    
    def test(self, loader):
        ensemble_pred = []
        for idx in range(self.M):
            network_pred = []
            for x,_ in loader:
                network_pred.append(self.network_list[idx](x.to(self.device)).detach())
            ensemble_pred.append(torch.cat(network_pred, dim=0))
        ensemble_pred = torch.stack(ensemble_pred)

        return ensemble_pred.detach() # M x N x C
    

def train_loop(loader, model, optimizer, nll_func, mse_func, scheduler=None, device='cpu'):
    nll = 0
    mse = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        y = y.reshape((-1,1))
        pred, var = model(x)
        loss = nll_func(pred, y, var)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        nll += loss.item()
        mse += mse_func(pred,y).item()

    nll /= len(loader)
    mse /= len(loader)
    return nll, mse