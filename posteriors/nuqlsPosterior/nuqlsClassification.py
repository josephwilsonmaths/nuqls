import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import copy
import time
from torch.profiler import profile, record_function, ProfilerActivity
import tracemalloc
import posteriors.nuqlsPosterior.nuqlsUtils as nqlutil

torch.set_default_dtype(torch.float64)


class classificationParallel(object):
    def __init__(self, network):
        self.network = network
        self.device = next(network.parameters()).device

    def train(self, train, train_bs, n_output, scale, S, epochs, lr, mu, verbose=False, extra_verbose=False):
        
        train_loader = DataLoader(train,train_bs)

        params = {k: v.detach() for k, v in self.network.named_parameters()}
        num_p = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        def fnet(params, x):
            return functional_call(self.network, params, x)

        p = copy.deepcopy(params).values()
        theta_S = torch.empty((num_p,S), device=self.device)

        def jvp_first(theta_s,params,x):
            dparams = nqlutil._dub(nqlutil.unflatten_like(theta_s, params.values()),params)
            _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                    (params,), (dparams,))
            proj = proj.detach()
            return proj

        def vjp_second(resid_s,params,x):
            _, vjp_fn = torch.func.vjp(lambda param: fnet(param, x), params)
            vjp = vjp_fn(resid_s)
            return vjp
        
        for s in range(S):
            theta_star = []
            for pi in p:
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=self.device)*scale + pi.to(self.device)))
            theta_S[:,s] = nqlutil.flatten(theta_star).detach()

        theta_S = theta_S.detach()
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        with torch.no_grad():
            for epoch in pbar:
                loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    x, y = x.to(device=self.device, non_blocking=True), y.to(device=self.device, non_blocking=True)
                    f_nlin = self.network(x)
                    proj = torch.vmap(jvp_first, (1,None,None))((theta_S),params,x).permute(1,2,0)
                    f_lin = (proj + f_nlin.unsqueeze(2))
                    Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=n_output)
                    resid = (Mubar - ybar.unsqueeze(2))
                    projT = torch.vmap(vjp_second, (2,None,None))(resid,params,x)
                    vjp = [j.detach().flatten(1) for j in projT[0].values()]
                    vjp = torch.cat(vjp,dim=1).detach()
                    g = (vjp.T / x.shape[0]).detach()

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g
                    theta_S -= lr*bt

                    l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    loss += l

                    a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    acc += a

                    if extra_verbose:
                        ma_l = loss / (pbar_inner.format_dict['n'] + 1)
                        ma_a = acc / ((pbar_inner.format_dict['n'] + 1) * x.shape[0])
                        metrics = {'min_loss_ma': ma_l.min().item(),
                                'max_loss_batch': ma_l.max().item(),
                                'min_acc_batch': ma_a.min().item(),
                                'max_acc_batch': ma_a.max().item(),
                                    'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device == torch.device('cuda'):
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        pbar_inner.set_postfix(metrics)


                loss /= len(train_loader)
                acc /= len(train)

                if verbose:
                    metrics = {'min_loss': loss.min().item(),
                               'max_loss': loss.max().item(),
                               'min_acc': acc.min().item(),
                               'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device == torch.device('cuda'):
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    pbar.set_postfix(metrics)

        if verbose:
            print('Posterior samples computed!')
        self.theta_S = theta_S

        return loss.max().item(), acc.min().item()
    
    def test(self, test, test_bs=50):
        params = {k: v.detach() for k, v in self.network.named_parameters()}

        test_loader = DataLoader(test, test_bs)

        def fnet(params, x):
            return functional_call(self.network, params, x)

        def jvp_first(theta_s,params,x):
            dparams = nqlutil._dub(nqlutil.unflatten_like(theta_s, params.values()),params)
            _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                    (params,), (dparams,))
            proj = proj.detach()
            return proj
        
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            proj = torch.vmap(jvp_first, (1,None,None))((self.theta_S),params,x).permute(1,2,0)
            f_lin = (proj + f_nlin.unsqueeze(2))
            pred_s.append(f_lin.detach())

        id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
        
        del f_lin
        del pred_s
    
        return id_predictions
    
    def UncertaintyPrediction(self, test, test_bs):

        logits = self.test(test, test_bs)

        probits = logits.softmax(dim=2)
        mean_prob = probits.mean(0)
        var_prob = probits.var(0)

        return mean_prob.detach(), var_prob.detach()
    