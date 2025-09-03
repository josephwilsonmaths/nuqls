import torch
from functorch import make_functional, make_functional_with_buffers
from torch.func import functional_call
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

class classificationParallel(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        print(f'NUQLS is using device {self.device}.')

        self.params = tuple(self.network.parameters())
        child_list = list(network.children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        final_layer = child_list[-1]
        self.llp = list(final_layer.parameters())[0].size().numel()
        self.num_output = list(final_layer.parameters())[0].shape[0]
        self.ck = torch.nn.Sequential(*first_layers)
        self.llp = list(self.ck.parameters())[-1].shape[0]
        self.theta_t = flatten(self.params)[-(self.llp*self.num_output+self.num_output):-self.num_output]

    def jvp(self, x, v):
        j = self.ck(x.to(device)).detach().reshape(x.shape[0], self.llp)
        X = v.reshape(self.num_output, self.llp, -1).permute(1,0,2)
        return torch.einsum('np,pcs->ncs', j,X).detach().to(device)
    
    def vjp(self, x,V):
        # V : x.shape[0] x self.n_output x self.S
        jt = self.ck(x.to(device)).detach().reshape(x.shape[0], self.llp).T
        return torch.einsum('pn,ncs->pcs',jt,V).detach().to(device)
    
    def train(self, train, batchsize, S, scale, epochs, lr, mu, verbose=False, extra_verbose=False, save_weights=None):
        train_loader = DataLoader(train,batchsize)

        self.theta = torch.randn(size=(self.llp*self.num_output,S),device=device)*scale + self.theta_t.unsqueeze(1)
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': scale,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        with torch.no_grad():
            for epoch in pbar:
                loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    x, y = x.to(device=self.device, non_blocking=True, dtype=torch.float64), y.to(device=self.device, non_blocking=True)
                    f_nlin = self.network(x)
                    f_lin = (self.jvp(x, (self.theta.to(device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,S)
                    Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=self.num_output)
                    g = self.vjp(x,(Mubar - ybar.unsqueeze(2))).permute(1,0,2).flatten(0,1) / x.shape[0]

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    self.theta -= lr*bt

                    l = (- 1 / x.shape[0] * torch.sum((ybar.flatten(0,1).unsqueeze(1) * torch.log(Mubar.flatten(0,1))),dim=0)).cpu()
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
                        if self.device.type == 'cuda':
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
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': self.theta,
                                'min_loss': loss.min().item(),
                                'max_loss': loss.max().item(),
                                'min_acc': acc.min().item(),
                                'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')

        return loss.max().item(), acc.min().item()

    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta_S = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            f_lin = (self.jvp(x, (self.theta.to(device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
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
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        f_lin = (self.jvp(x, (self.theta.to(device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
        return f_lin.detach().permute(2,0,1) 




class small_classification_parallel(object):
    '''
    NUQLS implementation for: smaller dataset/model classification, e.g. lenet5 on FMNIST. 
                        Larger S. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
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
        self.fnet, self.params, self.buffers = make_functional_with_buffers(self.net)
        self.theta_t = flatten(self.params).detach()  
        self.device = torch.device(device)

        if isinstance(self.net,model.resnet.ResNet):
            self.ck = self.net.ll
            self.llp = self.net.llp
        else:
            # first_layers = list(self.net.children())[:-1]
            first_layers = list(self.net.children())[0][:-1]
            self.ck = torch.nn.Sequential(*first_layers).to(device)
            # print(list(self.ck.parameters()))
            self.llp = list(self.ck.parameters())[-1].shape[0]
            # self.llp = list(self.ck.parameters()).shape[0]

        self.theta_t = flatten(self.params)[-(self.llp*self.n_output+self.n_output):-self.n_output]
        self.theta = torch.randn(size=(self.llp*self.n_output,self.S),device=device)*self.init_scale + self.theta_t.unsqueeze(1)


    def method(self,test,ood_test=None,mu=0,weight_decay=0,verbose=False, progress_bar = True, gradnorm=False):
        ## Create new parameter to train
        # self.theta = (torch.randn(size=(self.theta_t.shape[0],self.S),device=device)*self.init_scale + self.theta_t.unsqueeze(1)).detach()

        ## Create loaders and get entire training set
        total_train_loader = DataLoader(self.train,batch_size=len(self.train))
        X, Y = next(iter(total_train_loader))
        X, Y = X.to(device), Y.to(device)
        
        test_loader = DataLoader(test,batch_size=self.bs_test)
        ood_test_loader = DataLoader(ood_test, batch_size=self.bs_test)

        train_loader = DataLoader(self.train, batch_size=self.bs)
        
        ## Train S realisations of linearised networks
        if progress_bar:
            pbar = tqdm.trange(self.epochs)
        else:
            pbar = range(self.epochs)

        # J = jacobian(X[0:9])
        # print(J[:,0,-850:])
        # print(J[:,3,-850:])

        def jacobian(x):
            J = torch.zeros(x.shape[0],self.n_output,self.llp*self.n_output)
            v = self.ck(x.to(device)).detach().reshape(x.shape[0], self.llp)
            for c in range(self.n_output):
                J[:,c,c*self.llp:(c+1)*self.llp] = v
            return J.detach().to(device)
        
        def jvp(x, v):
            j = self.ck(x.to(device)).detach().reshape(x.shape[0], self.llp)
            X = v.reshape(self.n_output, self.llp, self.S).permute(1,0,2)
            return torch.einsum('np,pcs->ncs', j,X).detach().to(device)

        def vjp(x,V):
            # V : x.shape[0] x self.n_output x self.S
            jt = self.ck(x.to(device)).detach().reshape(x.shape[0], self.llp).T
            return torch.einsum('pn,ncs->pcs',jt,V).detach().to(device)
            
        # J = jacobian(X).detach().to(device)
        
        for epoch in pbar:
            loss = torch.zeros(self.S)
            accuracy = torch.zeros(self.S)
            with torch.no_grad():

                if verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    x, y = x.to(device), y.to(device)

                    # Jx = jvp(x, (self.theta.to(device) - self.theta_t.unsqueeze(1)))
                    # print(f'Jx.shape: {Jx.shape}')

                    # Jx_explicit = jacobian(x).flatten(0,1) @ (self.theta.to(device) - self.theta_t.unsqueeze(1))
                    # print(f'Jx_explicit.shape: {Jx_explicit.shape}') 

                    # print(torch.linalg.norm(Jx.flatten(0,1) - Jx_explicit))         

                    # import sys; sys.exit(0)     

                    f_nlin = self.net(x)
                    # f_lin = (jacobian(x).flatten(0,1) @ (self.theta.to(device) - self.theta_t.unsqueeze(1)) + 
                    #         f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S)
                    f_lin = (jvp(x, (self.theta.to(device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S)

                    # Mubar = nn.functional.softmax(f_lin,dim=1).flatten(0,1)
                    # ybar = torch.nn.functional.one_hot(y,num_classes=self.n_output).flatten(0,1)
                    # grad = (jacobian(x).flatten(0,1).T @ (Mubar - ybar.unsqueeze(1)) / x.shape[0])

                    Mubar = nn.functional.softmax(f_lin,dim=1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=self.n_output)
                    grad = vjp(x,(Mubar - ybar.unsqueeze(2))).permute(1,0,2).flatten(0,1) / x.shape[0]

                    loss += (- 1 / x.shape[0] * torch.sum((ybar.flatten(0,1).unsqueeze(1) * torch.log(Mubar.flatten(0,1))),dim=0)).cpu()

                    if epoch == 0:
                        bt = grad
                    else:
                        bt = mu*bt + grad + weight_decay * self.theta

                    self.theta -= self.lr * bt

                    accuracy += (f_lin.argmax(1) == y.unsqueeze(1)).type(torch.float).mean(dim=0).cpu()

                    if verbose:
                        ma_l = loss / (pbar_inner.format_dict['n'] + 1)
                        ma_a = accuracy / ((pbar_inner.format_dict['n'] + 1))
                        metrics = {'min_loss_ma': ma_l.min().item(),
                                'max_loss_batch': ma_l.max().item(),
                                'min_acc_batch': ma_a.min().item(),
                                'max_acc_batch': ma_a.max().item(),
                                    'resid_norm': torch.mean(torch.square(grad)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        pbar_inner.set_postfix(metrics)

                    if gradnorm:
                        total_grad = grad.detach()
                loss /= len(train_loader)
                accuracy /= len(train_loader)

            loss = torch.max(loss) # Report maximum CE loss over realisations.
            accuracy = torch.min(accuracy) # Report minimum CE loss over realisations.
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
            f_lin = (jacobian(x).flatten(0,1).to(device) @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S).detach()
            pred_s.append(f_lin.detach())

        id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C

        if ood_test is not None:
            pred_s = []
            for x,y in ood_test_loader:
                x, y = x.to(device), y.to(device)
                f_nlin = self.net(x)
                f_lin = (jacobian(x).flatten(0,1).to(device) @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S).detach()
                pred_s.append(f_lin.detach())

            ood_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
            return id_predictions, ood_predictions, res
        
        del f_lin
        del J
        del pred_s
    
        return id_predictions, res
    
class regressionParallel(object):
    '''
    NUQLS implementation for: regression, larger S, tiny dataset size. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,network):
        self.network = network
        self.device = next(network.parameters()).device

        child_list = list(self.network.children())
        if len(child_list) > 1:
            first_layers = child_list[:-1]
        elif len(child_list) == 1:
            first_layers = child_list[0][:-1]
        self.ck = torch.nn.Sequential(*first_layers)

        self.llp = list(self.ck.parameters())[-1].shape[0]

    def train(self, train, batchsize, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False):
        _, params = make_functional(self.network)
        theta_t = flatten(params)[-(self.llp+1):-1]
        theta = torch.randn(size=(self.llp,S),device=device)*scale + theta_t.unsqueeze(1)

        ## Create loaders and get entire training set
        train_loader = DataLoader(train,batch_size=batchsize)
    
        if batchsize == len(train):
            X,_ = next(iter(train_loader))
            J = self.ck(X.to(device)).detach()
        else:
            J = None

        def jacobian(x):
            if J is not None:
                return J
            else:
                return self.ck(x.to(device)).detach()

        # Set progress bar
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        ## Train S realisations of linearised networks
        for epoch in pbar:
            loss = 0
            resid_norm = 0

            if self.device == torch.device('cuda'):
                torch.cuda.reset_peak_memory_stats()

            for x,y in train_loader:
                x, y = x.to(self.device), y.to(self.device).reshape(-1,1)
                f_nlin = self.network(x)
                jacobian_batch = jacobian(x)
                f_lin = (jacobian_batch.to(self.device) @ (theta - theta_t.unsqueeze(1)) + f_nlin).detach()
                resid = f_lin - y
                g = jacobian_batch.T.to(self.device) @ resid.to(self.device) / x.shape[0]

                if epoch == 0:
                    bt = g
                else:
                    bt = mu*bt + g

                theta -= lr * bt

                loss += torch.mean(torch.square(jacobian_batch @ (theta - theta_t.unsqueeze(1)) + f_nlin - y)).max().item()
                resid_norm += torch.mean(torch.square(g)).item()

            loss /= len(train_loader)
            resid_norm /= len(train_loader)

            if verbose:
                metrics = {'max_loss': loss,
                   'resid_norm': resid_norm}
                if self.device == torch.device('cuda'):
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

        if verbose:
            print('Posterior samples computed!')

        self.theta = theta

        return loss, resid_norm
    
    def test(self, test, test_bs=1000):
        _, params = make_functional(self.network)
        theta_t = flatten(params)[-(self.llp+1):-1]

        test_loader = DataLoader(test,batch_size=test_bs)
                
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            J = self.ck(x.to(device)).detach()

            pred_lin = J @ (self.theta - theta_t.unsqueeze(1)) + f_nlin ## n x S
            pred_s.append(pred_lin.detach())

        predictions = torch.cat(pred_s,dim=0)
        return predictions
    
    def HyperparameterTuning(self, validation, left, right, its, verbose=False):
        val_predictions = self.test(validation)

        calibration_test_loader_val = DataLoader(validation,len(validation))
        _, val_y = next(iter(calibration_test_loader_val))

        left_scale, right_scale = left, right

        # Ternary Search
        for k in range(its):
            left_third = left_scale + (right_scale - left_scale) / 3
            right_third = right_scale - (right_scale - left_scale) / 3

            # Left ECE
            scaled_nuqls_predictions = val_predictions * left_third
            obs_map, predicted = calibration_curve_r(val_y,val_predictions.mean(1),scaled_nuqls_predictions.var(1),11)
            left_ece = torch.mean(torch.square(obs_map - predicted))

            # Right ECE
            scaled_nuqls_predictions = val_predictions * right_third
            obs_map, predicted = calibration_curve_r(val_y,val_predictions.mean(1),scaled_nuqls_predictions.var(1),11)
            right_ece = torch.mean(torch.square(obs_map - predicted))

            if left_ece > right_ece:
                left_scale = left_third
            else:
                right_scale = right_third

            scale = (left_scale + right_scale) / 2

            # Print info
            if verbose:
                print(f'\nSCALE {scale:.3}:: MAP: [{left_ece:.1%},{right_ece:.1%}]') 

            if abs(right_scale - left_scale) <= 1e-2:
                if verbose:
                    print('Converged.')
                break

        self.scale_cal = scale

    def eval(self,x):
        _, params = make_functional(self.network)
        theta_t = flatten(params)[-(self.llp+1):-1]

        x = x.to(self.device)
        f_nlin = self.network(x)
        J = self.ck(x.to(device)).detach()

        pred_lin = J @ (self.theta - theta_t.unsqueeze(1)) + f_nlin ## n x S
        return pred_lin.detach()
    




class regression_fullbatch(object):
    '''
    NUQLS implementation for: regression, larger S, tiny dataset size. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,net,train,S,epochs,lr,init_scale=1):
        self.net = net
        self.train = train
        self.init_scale = init_scale
        self.S = S
        self.epochs = epochs
        self.lr = lr
        self.fnet, self.params = make_functional(self.net)

        child_list = list(self.net.children())
        if len(child_list) > 1:
            first_layers = child_list[:-1]
        elif len(child_list) == 1:
            first_layers = child_list[0][:-1]
        self.ck = torch.nn.Sequential(*first_layers)

        self.llp = list(self.ck.parameters())[-1].shape[0]

        self.llp = list(self.ck.parameters())[-1].shape[0]
        self.theta_t = flatten(self.params)[-(self.llp+1):-1]
        self.theta = torch.randn(size=(self.llp,self.S),device=device)*self.init_scale + self.theta_t.unsqueeze(1)

        

    def train_linear(self,mu=0,weight_decay=0,my=0,sy=1,threshold=None,verbose=False, progress_bar=True):
        
        ## Create loaders and get entire training set
        train_loader_total = DataLoader(train,batch_size=len(train))
        X,Y = next(iter(train_loader_total))
        
        J = self.ck(X.to(device)).detach()

        # Set progress bar
        if progress_bar:
            pbar = tqdm.trange(self.epochs)
        else:
            pbar = range(self.epochs)
        
        ## Train S realisations of linearised networks
        for epoch in pbar:
            X, Y = X.to(device), Y.to(device).reshape(-1,1)
            
            f_nlin = self.net(X)
            f_lin = (J.to(device) @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin).detach()
            resid = f_lin - Y
            grad = J.T.to(device) @ resid.to(device) / X.shape[0] + weight_decay * self.theta

            if epoch == 0:
                bt = grad
            else:
                bt = mu*bt + grad

            self.theta -= self.lr * bt

            loss = torch.mean(torch.square(J @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin - Y)).max()

            if verbose:
                metrics = {'mean_loss': loss.item(),
                   'resid_norm': torch.mean(torch.square(grad)).item()}
                if self.device == torch.device('cuda'):
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

        # Report maximum loss over S, and the mean gradient norm
        max_l2_loss = torch.mean(torch.square(J @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin - Y)).max()
        norm_resid = torch.mean(torch.square(J.T @ ( f_nlin + J @ (self.theta - self.theta_t.unsqueeze(1)) - Y)))

        return max_l2_loss.detach(), norm_resid.detach()
    
    def test_linear(self,test,bs,my=0,sy=1):

        test_loader = DataLoader(test,batch_size=bs)
                
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            f_nlin = self.net(x)
            J = self.ck(x).detach()

            pred_lin = J @ (self.theta - self.theta_t.unsqueeze(1)) + f_nlin ## n x S
            pred_s.append(pred_lin.detach()*sy + my)

        predictions = torch.cat(pred_s,dim=0)
        return predictions


def series_method(net, train_data, test_data, ood_test_data=None, regression = False, scheduler_true = False, train_bs = 100, test_bs = 100, S = 10, scale=1, lr=1e-3, epochs=20, mu=0.9, wd = 0, verbose=False, progress_bar = True):
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
        if scheduler_true:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs)

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
            if scheduler_true:
                scheduler.step()
            
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
            print(f'loss = {loss_e}')
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

def regression_parallel(net, train, test, ood_test=None, train_bs=50, test_bs=50, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9):
    fnet, params = make_functional(net)
    theta_t = flatten(params)

    num_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    train_loader = DataLoader(train,train_bs)
    test_loader = DataLoader(test,batch_size=test_bs)
    if ood_test is not None:
        ood_test_loader = DataLoader(ood_test,batch_size=test_bs)

    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)

    p = copy.deepcopy(params)
    theta_S = torch.empty((num_p,S), device=device)

    for s in range(S):
        theta_star = []
        for pi in p:
            theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=device)*scale + pi.to(device)))
        theta_S[:,s] = flatten(theta_star)

    def jvp_first(theta_s,params,x):
        dparams = _sub(tuple(unflatten_like(theta_s, params)),params)
        _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                (params,), (dparams,))
        return proj

    def vjp_second(resid_s,params,x):
        _, vjp_fn = torch.func.vjp(lambda param: fnet(param, x), params)
        vjp = vjp_fn(resid_s.unsqueeze(1))
        return vjp
    
    print(f'mem before = {(1e-9*torch.cuda.memory_allocated()):.4}')

    for epoch in tqdm.trange(epochs):
        torch.cuda.reset_peak_memory_stats()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            f_nlin = net(x)
            proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).flatten(1).T
            f_lin = (proj + f_nlin).detach()
            resid = f_lin - y
            projT = torch.vmap(vjp_second, (1,None,None))(resid,params,x)
            vjp = [j.detach().flatten(1) for j in projT[0]]
            vjp = torch.cat(vjp,dim=1).detach()
            g = vjp.T / x.shape[0]

            if epoch == 0:
                bt = g
            else:
                bt = mu*bt + g

            theta_S -= lr*bt

        if epoch % 1 == 0:
            print("\n-----------------")
            print("Epoch {} of {}".format(epoch,epochs))
            print("max l2 loss = {:.4}".format(torch.mean(torch.square(resid)).max()))
            print("Residual of normal equation l2 = {:.4}".format(torch.mean(torch.square(g))))
            print(f'Max Mem used = {(1e-9*torch.cuda.max_memory_allocated()):.3} gb')


    # Concatenate predictions
    pred_test = []
    for x,_ in test_loader:
        x = x.to(device)
        f_nlin = net(x)
        proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).flatten(1).T
        pred = proj + f_nlin
        pred_test.append(pred.detach())

    id_predictions = torch.cat(pred_test,dim=0)

    if ood_test is not None:
        pred_test = []
        for x,_ in ood_test_loader:
            x = x.to(device)
            f_nlin = net(x)
            proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).flatten(1).T
            pred = proj + f_nlin
            pred_test.append(pred.detach())

        ood_predictions = torch.cat(pred_test,dim=0)

    if ood_test is not None:
      return id_predictions, ood_predictions

    return id_predictions

class classification_ll(object):
    '''
    NUQLS implementation for: smaller dataset/model classification, e.g. lenet5 on FMNIST. 
                        Larger S. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
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
        self.theta_t = flatten(self.params).detach()[-850:-10]  

    def method(self,test,ood_test=None,mu=0,weight_decay=0,verbose=False, progress_bar = True, gradnorm=False):
        ## Create new parameter to train
        self.theta = (torch.randn(size=(840,self.S),device=device)*self.init_scale + self.theta_t.unsqueeze(1)).detach()

        ## Create loaders and get entire training set
        total_train_loader = DataLoader(self.train,batch_size=len(self.train))
        X, Y = next(iter(total_train_loader))
        X, Y = X.to(device), Y.to(device)
        test_loader = DataLoader(test,batch_size=self.bs_test)
        ood_test_loader = DataLoader(ood_test, batch_size=self.bs_test)

        ## Compute jacobian of net, evaluated on training set
        def jacobian(x):
            J = self.net.ll(x)
            return J
        
        ## Train S realisations of linearised networks
        if progress_bar:
            pbar = tqdm.trange(self.epochs)
        else:
            pbar = range(self.epochs)

        J = jacobian(X)
        print(J.shape)
        
        for epoch in pbar:
            loss = torch.zeros(self.S)
            accuracy = torch.zeros(self.S)
            with torch.no_grad():
                f_nlin = self.net(x)
                f_lin = (J.flatten(0,1) @ (self.theta.to(device) - self.theta_t.unsqueeze(1)) + 
                        f_nlin.reshape(-1,1)).reshape(x.shape[0],self.n_output,self.S)
                Mubar = nn.functional.softmax(f_lin,dim=1).flatten(0,1)
                ybar = torch.nn.functional.one_hot(y,num_classes=self.n_output).flatten(0,1)
                grad = (J.flatten(0,1).T @ (Mubar - ybar.unsqueeze(1)) / x.shape[0])

                loss = - 1 / x.shape[0] * torch.sum((ybar.unsqueeze(1) * torch.log(Mubar)),dim=0)

                if epoch == 0:
                    bt = grad
                else:
                    bt = mu*bt + grad + weight_decay * self.theta

                self.theta -= self.lr * bt

                accuracy = (f_lin.argmax(1) == y.unsqueeze(1)).type(torch.float).mean(dim=0)
                if gradnorm:
                    total_grad = grad.detach()

            loss = torch.max(loss) # Report maximum CE loss over realisations.
            accuracy = torch.min(accuracy) # Report minimum CE loss over realisations.
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


# def classification_parallel_i(net, train, test, ood_test=None, train_bs=50, test_bs=50, n_output = 10, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9):
#     # fnet, params = make_functional(net)
#     # theta_t = flatten(params)

#     # Detaching the parameters because we won't be calling Tensor.backward().
#     params = {k: v.detach() for k, v in net.named_parameters()}

#     def fnet(params, x):
#         return functional_call(net, params, x)

#     num_p = sum(p.numel() for p in net.parameters() if p.requires_grad)

#     train_loader = DataLoader(train,train_bs,num_workers=0, pin_memory=False, pin_memory_device=device)
#     test_loader = DataLoader(test,batch_size=test_bs)
#     if ood_test is not None:
#         ood_test_loader = DataLoader(ood_test,batch_size=test_bs)

#     def fnet_single(params, x):
#         return fnet(params, x.unsqueeze(0)).squeeze(0)

#     p = copy.deepcopy(params).values()
#     theta_S = torch.empty((num_p,S), device=device)

#     for s in range(S):
#         theta_star = []
#         for pi in p:
#             theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=device)*scale + pi.to(device)))
#         theta_S[:,s] = flatten(theta_star).detach()

#     theta_S = theta_S.detach()

#     def jvp_first(theta_s,params,x):
#         dparams = _dub(unflatten_like(theta_s, params.values()),params)
#         _, proj = torch.func.jvp(lambda param: fnet(param, x),
#                                 (params,), (dparams,))
#         proj = proj.detach()
#         return proj

#     # def jvp_first(jvp,theta_s,params):
#     #     dparams = _dub(unflatten_like(theta_s, params.values()),params)
#     #     proj = jvp((dparams,)).detach()
#     #     return proj

#     def vjp_second(resid_s,params,x):
#         _, vjp_fn = torch.func.vjp(lambda param: fnet(param, x), params)
#         vjp = vjp_fn(resid_s)
#         return vjp
    

#     def jvp_first_old(theta_s,params,x):
#         torch.cuda.synchronize(device)
#         t = time.time()
#         dparams = _sub(tuple(unflatten_like(theta_s, params)),params)
#         t1 = time.time() - t
#         print(f't1 = {t1}')

#         torch.cuda.synchronize(device)
#         t = time.time()
#         _, proj = torch.func.jvp(lambda param: fnet(param, x),
#                                 (params,), (dparams,))
#         t2 = time.time() - t
#         print(f't2 = {t2}')
#         return proj

#     def vjp_second_old(resid_s,params,x):
#         _, vjp_fn = torch.func.vjp(lambda param: fnet(param, x), params)
#         vjp = vjp_fn(resid_s)
#         return vjp
    
#     # print(f'mem before = {(1e-9*torch.cuda.memory_allocated()):.4}')
#     # tracemalloc.start()
#     # current, peak =  tracemalloc.get_traced_memory()
#     # print(f"{current:0.2f}, {peak:0.2f}")
#     with torch.no_grad():
#         for epoch in tqdm.trange(epochs):
#             loss = 0
            
#             # tload = time.time()
#             i = 0
#             for x,y in train_loader:
#                 x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
#                 f_nlin = net(x)
#                 proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).permute(1,2,0)
#                 f_lin = (proj + f_nlin.unsqueeze(2))
#                 Mubar = nn.functional.softmax(f_lin,dim=1)
#                 ybar = torch.nn.functional.one_hot(y,num_classes=n_output)
#                 resid = (Mubar - ybar.unsqueeze(2))
#                 projT = torch.vmap(vjp_second, (2,None,None))(resid,params,x)
#                 vjp = [j.detach().flatten(1) for j in projT[0].values()]
#                 vjp = torch.cat(vjp,dim=1).detach()
#                 g = (vjp.T / x.shape[0]).detach()

#                 if epoch == 0:
#                     bt = g
#                 else:
#                     bt = mu*bt + g
#                 theta_S -= lr*bt

#                 l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach()
#                 # print(l)
#                 loss += l

#                 # print(f'Max Mem used = {(1e-9*torch.cuda.max_memory_allocated()):.3} gb')
#                     # print("Residual of normal equation l2 = {:.4}".format(torch.mean(torch.square(g))))
#             loss /= len(train_loader)        
#             # if epoch % 1 == 0:
#             #     print("\n-----------------")
#             #     print("Epoch {} of {}".format(epoch,epochs))
#             #     print("max ce loss = {:.4}".format(loss.max()))
#             #     print("Residual of normal equation l2 = {:.4}".format(torch.mean(torch.square(g))))
#             #     print(f'Max Mem used = {(1e-9*torch.cuda.max_memory_allocated()):.3} gb')
    
#     # Concatenate predictions
#         pred_s = []
#         for x,y in test_loader:
#             x, y = x.to(device), y.to(device)
#             f_nlin = net(x)
#             proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).permute(1,2,0)
#             f_lin = (proj + f_nlin.unsqueeze(2))
#             pred_s.append(f_lin.detach())

#         id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C

#         if ood_test is not None:
#             pred_s = []
#             for x,y in ood_test_loader:
#                 x, y = x.to(device), y.to(device)
#                 f_nlin = net(x)
#                 proj = torch.vmap(jvp_first, (1,None,None))(theta_S,params,x).permute(1,2,0)
#                 f_lin = (proj + f_nlin.unsqueeze(2))
#                 pred_s.append(f_lin.detach())

#             ood_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
#             return id_predictions, ood_predictions, loss.max()
        
#         del f_lin
#         del pred_s
    
#         return id_predictions, loss.max()
 
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

import scipy
def calibration_curve_r(target,mean,variance,c):
    target = target.detach().cpu(); mean = mean.detach().cpu(); variance = variance.detach().cpu()
    predicted_conf = torch.linspace(0,1,c)
    observed_conf = torch.empty((c))
    for i,ci in enumerate(predicted_conf):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_l = mean.reshape(-1) - z*torch.sqrt(variance.reshape(-1))
        ci_r = mean.reshape(-1) + z*torch.sqrt(variance.reshape(-1)) 
        observed_conf[i] = torch.logical_and(ci_l < target.reshape(-1), target.reshape(-1) < ci_r).type(torch.float).mean()
    return observed_conf,predicted_conf
