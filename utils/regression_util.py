import torch
import scipy
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

## Calibration function (ECE)
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

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight,mean=0,std=1)
        torch.nn.init.normal_(m.bias,mean=0,std=1)


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



class EnsembleNetwork(torch.nn.Module):
    def __init__(self, hidden_sizes, input_start, input_dim, activation='tanh'):
        super().__init__()
        self.ll = []
        self.hidden_sizes = hidden_sizes
        if len(hidden_sizes) > 1:
            for i,h in enumerate(hidden_sizes):
                if i == 0:
                    self.ll.append(torch.nn.Linear(input_dim-input_start,hidden_sizes[i]))
                else:
                    self.ll.append(torch.nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
        else:
            self.ll.append(torch.nn.Linear(input_dim-input_start,hidden_sizes[0]))
        self.ml = torch.nn.ModuleList(self.ll)
        if activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'relu':
            self.act = torch.nn.ReLU()
        self.linear_mu = torch.nn.Linear(hidden_sizes[-1],1)
        self.linear_sig = torch.nn.Linear(hidden_sizes[-1],1)

    def forward(self, x):
        for i in range(len(self.hidden_sizes)):
            x = self.act(self.ml[i](x))
        mu = self.linear_mu(x)
        variance = self.linear_sig(x)
        variance = torch.nn.functional.softplus(variance) + 1e-6
        return mu, variance

class CustomNLL(torch.nn.Module):
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

## Possible new networks

# class ResRegressionDNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         if args.activation == 'tanh':
#             self.act = nn.Tanh()
#         elif args.activation == 'relu':
#             self.act = nn.ReLU()
#         self.w1 = 100
#         self.w2 = 128
#         # Block 1
#         self.linear1 = nn.Linear(input_dim-input_start,self.w1)
#         self.linear2 = nn.Linear(self.w1,self.w2)
#         self.linear3 = nn.Linear(self.w2,input_dim-input_start)
#         # Block 2
#         self.linear4 = nn.Linear(input_dim-input_start,self.w1)
#         self.linear5 = nn.Linear(self.w1,self.w2)
#         self.linear6 = nn.Linear(self.w2,input_dim-input_start)
#         # Block 3
#         self.linear7 = nn.Linear(input_dim-input_start,self.w1)
#         self.linear8 = nn.Linear(self.w1,self.w2)
#         self.linear9 = nn.Linear(self.w2,input_dim-input_start)
#         # Final
#         self.linear10 = nn.Linear(input_dim-input_start,self.w1)
#         self.linear11 = nn.Linear(self.w1,self.w2)
#         self.linear12 = nn.Linear(self.w2,1)

#     def forward(self,x):
#         # Block 1
#         initial = x
#         x = self.act(self.linear1(x))
#         x = self.act(self.linear2(x))
#         x = self.act(self.linear3(x))
#         x = self.act(x + initial)
#         # Block 2
#         initial = x
#         x = self.act(self.linear4(x))
#         x = self.act(self.linear5(x))
#         x = self.act(self.linear6(x))
#         x = self.act(x + initial)
#         # Block 3
#         initial = x
#         x = self.act(self.linear7(x))
#         x = self.act(self.linear8(x))
#         x = self.act(self.linear9(x))
#         x = self.act(x + initial)
#         # Final
#         x = self.act(self.linear10(x))
#         x = self.act(self.linear11(x))
#         x = self.act(self.linear12(x))

#         return x

# class LargeDNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.nb = 3
#         self.ll = []
#         if len(hidden_sizes) > 1:
#             for i,h in enumerate(hidden_sizes):
#                 if i == 0:
#                     self.ll.append(nn.Linear(input_dim-input_start,hidden_sizes[i]))
#                 else:
#                     self.ll.append(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
#         else:
#             self.ll.append(nn.Linear(input_dim-input_start,hidden_sizes[0]))
#         self.ml = nn.ModuleList(self.ll)
#         if args.activation == 'tanh':
#             self.act = nn.Tanh()
#         elif args.activation == 'relu':
#             self.act = nn.ReLU()
#         self.block_end = nn.Linear(hidden_sizes[-1],input_dim-input_start)
#         self.linear_end = nn.Linear(hidden_sizes[-1],1)

#     def forward(self, x):
#         for j in range(self.nb):
#             initial = x
#             for i in range(len(hidden_sizes)):
#                 x = self.act(self.ml[i](x))
#             x  = self.act(x + initial)
#         for i in range(len(hidden_sizes)):
#             x = self.act(self.ml[i](x))
#         x = self.linear(x)
#         return x

from functorch import make_functional
from torch.func import functional_call, vmap, jacrev, jvp

def rank_calc(net, train):

    # Compute NTKGP
    fnet, params = make_functional(net)

    ## Compute jacobian of net, evaluated on training set
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)

    def Jx(Xs):
        J = vmap(jacrev(fnet_single), (None, 0))(params, Xs)
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1).detach()
        return J

    Jtrain = Jx(train)
    ntk_shape = Jtrain.shape
    ntk_linear_dependence = torch.linalg.norm(torch.eye(Jtrain.shape[1]) - torch.linalg.pinv(Jtrain) @ Jtrain)
    ntk_rank = torch.linalg.matrix_rank(Jtrain)

    def Jx(Xs):
        J = vmap(jacrev(fnet_single), (None, 0))(params, Xs)
        # J = [j.detach().flatten(1) for j in J]
        # J = torch.cat(J,dim=1).detach()
        J = J[-2].detach().flatten(1).detach() # For last-layer test
        return J

    Jtrain = Jx(train)
    ck_linear_dependence = torch.linalg.norm(torch.eye(Jtrain.shape[1]) - torch.linalg.pinv(Jtrain) @ Jtrain)
    ck_rank = torch.linalg.matrix_rank(Jtrain)
    ck_shape= Jtrain.shape
    return ntk_rank, ntk_linear_dependence, ntk_shape, ck_rank, ck_linear_dependence, ck_shape