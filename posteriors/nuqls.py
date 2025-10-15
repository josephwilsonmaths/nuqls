import torch
from functorch import make_functional, make_functional_with_buffers
from torch.func import functional_call, vmap, jacrev, jvp
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import copy
import models as model
import time
from torch.profiler import profile, record_function, ProfilerActivity
import tracemalloc

torch.set_default_dtype(torch.float64)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if device=='cuda':
    torch.backends.cudnn.benchmark = True

class classification_parallel_i(object):
    def __init__(self, net):
        self.net = net

    def train(self, train, train_bs=50, n_output = 10, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False, progress_bar=True):
        
        train_loader = DataLoader(train,train_bs,num_workers=0, pin_memory=False, pin_memory_device=device)

        params = {k: v.detach() for k, v in self.net.named_parameters()}
        num_p = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        def fnet(params, x):
            return functional_call(self.net, params, x)

        p = copy.deepcopy(params).values()
        theta_S = torch.empty((num_p,S), device=device)

        def jvp_first(theta_s,params,x):
            dparams = _dub(unflatten_like(theta_s, params.values()),params)
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
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=device)*scale + pi.to(device)))
            theta_S[:,s] = flatten(theta_star).detach()

        theta_S = theta_S.detach()
        
        if progress_bar:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        with torch.no_grad():
            for epoch in pbar:
                loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')
                i = 0
                for x,y in train_loader:
                    x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
                    f_nlin = self.net(x)
                    proj = torch.vmap(jvp_first, (1,None,None))((theta_S),params,x).permute(1,2,0)
                    f_lin = (proj + f_nlin.unsqueeze(2))
                    Mubar = torch.clamp(nn.functional.softmax(f_lin,dim=1),1e-32,1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=n_output)
                    resid = (Mubar - ybar.unsqueeze(2))
                    projT = torch.vmap(vjp_second, (2,None,None))(resid,params,x)
                    vjp = [j.detach().flatten(1) for j in projT[0].values()]
                    vjp = torch.cat(vjp,dim=1).detach()
                    g = (vjp.T / x.shape[0]).detach()

                    # if epoch < 6:
                    #     lr_sc = lr
                    # else:
                    #     lr_sc = lr / 10

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g
                    theta_S -= lr*bt

                    l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    loss += l

                    a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    acc += a


                loss /= len(train_loader)
                acc /= len(train)
                if verbose:        
                    if epoch % 1 == 0:
                        print("\n-----------------")
                        print("Epoch {} of {}".format(epoch+1,epochs))
                        print("min ce loss = {:.4}".format(loss.min()))
                        print("max ce loss = {:.4}".format(loss.max()))
                        print("min acc = {:.4}".format(acc.min()))
                        print("max acc = {:.4}".format(acc.max()))
                        print("Residual of normal equation l2 = {:.4}".format(torch.mean(torch.square(g))))
                        print(f'Max Mem used = {(1e-9*torch.cuda.max_memory_allocated()):.3} gb')
        
        if verbose:
            print('Posterior samples computed!')
        self.theta_S = theta_S

        return loss.max().item(), acc.min().item()
    
    def test(self, test, test_bs=50):
        params = {k: v.detach() for k, v in self.net.named_parameters()}

        test_loader = DataLoader(test, test_bs)

        def fnet(params, x):
            return functional_call(self.net, params, x)

        def jvp_first(theta_s,params,x):
            dparams = _dub(unflatten_like(theta_s, params.values()),params)
            _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                    (params,), (dparams,))
            proj = proj.detach()
            return proj
        
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            f_nlin = self.net(x)
            proj = torch.vmap(jvp_first, (1,None,None))((self.theta_S),params,x).permute(1,2,0)
            f_lin = (proj + f_nlin.unsqueeze(2))
            pred_s.append(f_lin.detach())

        id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
        
        del f_lin
        del pred_s
    
        return id_predictions

def linearize(f, params):
  def f_lin(p, *args, **kwargs):
    dparams = _sub(p, params)
    f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
                           (params,), (dparams,))
    return f_params_x + proj
  return f_lin

def _sub(x, y):
    return tuple(x - y for (x, y) in zip(x, y))

def _dub(x,y):
    return {yi:xi - y[yi] for (xi, yi) in zip(x, y)}

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return tuple(outList)
