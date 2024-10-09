import torch
from functorch import make_functional, make_functional_with_buffers
from torch.func import vmap, jacrev, jvp
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import copy

torch.set_default_dtype(torch.float64)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class classification_parallel(object):
    '''
    NUQLS implementation for: classification, larger S. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,net,train,S,epochs,lr,bs,bs_test,n_output,init_scale=1):
        self.net = net
        self.train = train
        self.init_scale = init_scale
        self.S = S
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        self.bs_test = bs_test
        self.n_output = n_output
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.fnet, self.params = make_functional(self.net)
        self.theta_t = flatten(self.params).detach()  

    def method(self,test,ood_test=None,mu=0,weight_decay=0,verbose=False, progress_bar = True, gradnorm=False):
        ## Create new parameter to train
        self.theta = (torch.randn(size=(self.theta_t.shape[0],self.S),device=device)*self.init_scale + self.theta_t.unsqueeze(1)).detach()

        ## Create loaders and get entire training set
        train_loader = DataLoader(self.train,batch_size=self.bs)
        test_loader = DataLoader(test,batch_size=self.bs_test)
        ood_test_loader = DataLoader(ood_test, batch_size=self.bs_test)

        ## Compute jacobian of net, evaluated on training set
        def jacobian(X):
            def fnet_single(params, x):
                return self.fnet(params, x.unsqueeze(0)).squeeze(0)
            J = vmap(jacrev(fnet_single), (None, 0))(self.params, X.to(device))
            J = [j.detach().flatten(2) for j in J]
            J = torch.cat(J,dim=2).detach()
            return J
        
        ## Train S realisations of linearised networks
        if progress_bar:
            pbar = tqdm.trange(self.epochs)
        else:
            pbar = range(self.epochs)
        
        for epoch in pbar:
            loss = torch.zeros(self.S)
            loss2 = 0
            accuracy = torch.zeros(self.S)
            if gradnorm:
                total_grad = 0
            for x,y in train_loader:
                x, y = x.to(device), y.to(device)
                J = jacobian(x).detach()
                with torch.no_grad():
                    f_nlin = self.net(x)
                    f_lin = (J.flatten(0,1) @ (self.theta.to(device) - self.theta_t.unsqueeze(1)) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S)
                    Mubar = nn.functional.softmax(f_lin,dim=1).flatten(0,1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=self.n_output).flatten(0,1)
                    grad = (J.flatten(0,1).T @ (Mubar - ybar.unsqueeze(1)) / x.shape[0])

                    loss -= 1 / x.shape[0] * torch.sum((ybar.unsqueeze(1) * torch.log(Mubar)),dim=0)

                    if epoch == 0:
                        bt = grad
                    else:
                        bt = mu*bt + grad + weight_decay * self.theta

                    self.theta -= self.lr * bt

                    accuracy += (f_lin.argmax(1) == y.unsqueeze(1)).type(torch.float).mean(dim=0)
                    if gradnorm:
                        total_grad += grad.detach()

            loss /= len(train_loader)
            loss = torch.max(loss) # Report maximum CE loss over realisations.
            accuracy /= len(train_loader)
            accuracy = torch.min(accuracy) # Report minimum CE loss over realisations.
            if gradnorm:
                total_grad /= len(train_loader)

            if epoch % 1 == 0 and verbose:
                print("\n-----------------")
                print("Epoch {} of {}".format(epoch,self.epochs))
                print("CE loss = {}".format(loss))
                print(f"Acc = {accuracy:.1%}")
                if gradnorm:
                    print("Max ||grad||_2 = {:.6f}".format(torch.max(torch.linalg.norm(total_grad,dim=0))))

        res = {'loss': loss,
               'acc': accuracy}
        if gradnorm:
            res['grad'] = torch.max(torch.linalg.norm(total_grad,dim=0))
        
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            f_nlin = self.net(x)
            J = jacobian(x)
            f_lin = (J.flatten(0,1).to(device) @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S).detach()
            pred_s.append(f_lin.detach())

        id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C

        if ood_test is not None:
            pred_s = []
            for x,y in ood_test_loader:
                x, y = x.to(device), y.to(device)
                f_nlin = self.net(x)
                J = jacobian(x)
                f_lin = (J.flatten(0,1).to(device) @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S).detach()
                pred_s.append(f_lin.detach())

            ood_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
            return id_predictions, ood_predictions, res
        
        del f_lin
        del J
        del pred_s
    
        return id_predictions, res

    
class small_regression_parallel(object):
    '''
    NUQLS implementation for: regression, larger S, tiny dataset size. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,net,train,S,epochs,lr,bs,bs_test,init_scale=1):
        self.net = net
        self.train = train
        self.init_scale = init_scale
        self.S = S
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        self.bs_test = bs_test
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.fnet, self.params = make_functional(self.net)
        self.theta_t = flatten(self.params)
        

    def method(self,test,mu=0,weight_decay=0,my=0,sy=1,verbose=False):
        ## Create new parameter to train
        self.theta = torch.randn(size=(self.theta_t.shape[0],self.S),device=device)*self.init_scale + self.theta_t.unsqueeze(1)
        self.theta_init = self.theta.clone().detach()

        ## Create loaders and get entire training set
        train_loader = DataLoader(self.train,batch_size=self.bs)
        train_loader_total = DataLoader(self.train,batch_size=len(self.train))
        test_loader = DataLoader(test,batch_size=self.bs_test)
        X,Y = next(iter(train_loader_total))
        f_diff = torch.empty((self.epochs))
        param_diff = torch.empty((self.epochs))

        ## Compute jacobian of net, evaluated on training set
        def fnet_single(params, x):
            return self.fnet(params, x.unsqueeze(0)).squeeze(0)
        J = vmap(jacrev(fnet_single), (None, 0))(self.params, X.to(device))
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()
        
        ## Train S realisations of linearised networks
        for epoch in range(self.epochs):
            X, Y = X.to(device), Y.to(device)

            f_nlin = self.net(X)
            f_lin = (J.to(device) @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin).detach()
            resid = f_lin - Y.unsqueeze(1)
            grad = J.T.to(device) @ resid.to(device) / X.shape[0] + weight_decay * self.theta

            if epoch == 0:
                bt = grad
            else:
                bt = mu*bt + grad

            self.theta -= self.lr * bt

            if epoch % 100 == 0 and verbose:
                print("\n-----------------")
                print("Epoch {} of {}".format(epoch,self.epochs))
                print("max l2 loss = {}".format(torch.mean(torch.square(J @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin - Y.unsqueeze(1))).max()))
                print("Residual of normal equation l2 = {}".format(torch.mean(torch.square(J.T @ ( f_nlin + J @ (self.theta - self.theta_t.unsqueeze(1)) - Y.unsqueeze(1))))))

        # Report maximum loss over S, and the mean gradient norm
        max_l2_loss = torch.mean(torch.square(J @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin - Y.unsqueeze(1))).max()
        norm_resid = torch.mean(torch.square(J.T @ ( f_nlin + J @ (self.theta - self.theta_t.unsqueeze(1)) - Y.unsqueeze(1))))
                
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            f_nlin = self.net(x)
            J = vmap(jacrev(fnet_single), (None, 0))(self.params, x)
            J = [j.detach().flatten(1) for j in J]
            J = torch.cat(J,dim=1)

            pred_lin = J @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin ## n x S
            pred_s.append(pred_lin.detach()*sy + my)

        predictions = torch.cat(pred_s,dim=1)
        return predictions, max_l2_loss, norm_resid

def series_method(net, train_data, test_data, ood_test_data=None, regression = False, train_bs = 100, test_bs = 100, S = 10, scale=1, lr=1e-3, epochs=20, mu=0.9, wd = 0, verbose=False, progress_bar = True):
    '''
    NUQLS implementation for: either regression/classification, small S. 
    Per epoch training is much faster compared to parallel implement, overall speed is faster when S <= 10.
    '''
    # Create functional version of net
    fnet, params, buffers = make_functional_with_buffers(net)

    # Create copy of trained parameters
    p = copy.deepcopy(params)
    
    # Turn off gradient tracking for theta_t and original net
    for pi in params:
        pi.requires_grad = False

    for m in fnet.modules():
      for pi in m.parameters():
          pi.requires_grad = False
          pi = pi.detach()

    # Create linearized network function
    f_lin = linearize(fnet, params)

    # Set progress bar
    if progress_bar:
        pbar = tqdm.trange(S)
    else:
        pbar = range(S)

    # Create dataloaders
    train_loader = DataLoader(train_data,batch_size=train_bs)
    test_loader = DataLoader(test_data,batch_size=test_bs)
    if ood_test_data is not None:
      ood_test_loader = DataLoader(ood_test_data,batch_size=test_bs)

    # Training result
    res = {'loss': 0,
        'acc': 1}
    
    # Create lists to save predictions into
    id_predictions = []
    ood_predictions = []

    for s in pbar:
        # Create new theta_star_0
        theta_star = []
        for pi in p:
            theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=device)*scale + pi.to(device)))
        theta_star = tuple(theta_star)

        # Training    
        optim = torch.optim.SGD(theta_star,lr=lr,momentum=mu,weight_decay=wd)
        if not regression:
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            loss_e = 0
            acc_e = 0
            for x,y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                pred = f_lin(theta_star, buffers, x)
                loss = loss_fn(pred, y)
                loss.backward()
                loss_e += loss
                if not regression:
                    acc_e += (pred.argmax(1) == y).type(torch.float).sum().item()
                optim.step()
            
            loss_e /= len(train_loader)
            if not regression:
                acc_e /= len(train_loader.dataset)
            if verbose:
                print(f"\nepoch: {epoch}")
                print(f"loss: {loss_e}")
                if not regression:
                    print(f"acc: {acc_e:.1%}")

        if verbose:
            print(f"\ns = {s}")
            print(f"loss: {loss_e}")
            if not regression:
                print(f"acc: {acc_e:.1%}\n")

        if loss_e > res["loss"]:
            res['loss'] = loss_e
        if acc_e < res['acc']:
            res['acc'] = acc_e

        # Concatenate predictions
        pred_test = []
        for x,_ in test_loader:
            x = x.to(device)
            pred = f_lin(theta_star, buffers, x)
            pred_test.append(pred.detach())

        id_predictions.append(torch.cat(pred_test,dim=0))

        if ood_test_data is not None:
            pred_test = []
            for x,_ in ood_test_loader:
                x = x.to(device)
                pred = f_lin(theta_star, buffers, x)
                pred_test.append(pred.detach())

            ood_predictions.append(torch.cat(pred_test,dim=0))

    id_predictions = torch.stack(id_predictions,dim=0)
    if ood_test_data is not None:
      ood_predictions = torch.stack(ood_predictions,dim=0)
      return id_predictions, ood_predictions, res

    return id_predictions, res
 
def linearize(f, params):
  def f_lin(p, *args, **kwargs):
    dparams = _sub(p, params)
    f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
                           (params,), (dparams,))
    return f_params_x + proj
  return f_lin

def _sub(x, y):
  return tuple(x - y for (x, y) in zip(x, y))

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
