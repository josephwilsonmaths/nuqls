import torch
from torch.func import functional_call
import tqdm
import sys

def l_layer_params(net):
    theta_star_k = []
    sd = net.state_dict()
    sdk = sd.keys()
    for i,p in enumerate(sdk):
        if i < len(sdk) - 2:
            theta_star_k.append(sd[p].flatten(0))
        else:
            theta_star_k.append(torch.zeros(sd[p].flatten(0).shape))
    theta_star_k = torch.cat(theta_star_k)
    return theta_star_k

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def train(dataloader, model, optimizer, loss_function, scheduler=None):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(dataloader):
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

def train_bde(train_loader,net,delta,theta_k,loss_fn,Lambda,optim,sched):
    train_loss = 0
    for i, (X,y) in enumerate(train_loader):
        # Get and prepare inputs
        y = y.reshape(-1,1)
        # Compute prediction error
        pred = net(X)
        

        # Add delta function to outputs
        pred = pred + delta(X)

        # Calculate loss
        loss = loss_fn(y, pred)

        # Regularisation
        theta_t = torch.nn.utils.parameters_to_vector(net.parameters())
        diff = theta_t - theta_k
        reg = diff @ (Lambda * diff)
        loss = 0.5 * loss + 0.5 * reg

        # Backpropagation
        loss.backward()

        optim.step()
        if sched is not None:
            sched.step()
        optim.zero_grad()

        train_loss += loss.item()

    train_loss = train_loss / (i+1)
    return train_loss

def bde_weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.normal_(m.weight,mean=0,std=1)
        torch.nn.init.normal_(m.bias,mean=0,std=1)

def _dub(x,y):
        return {yi:xi - y[yi] for (xi, yi) in zip(x, y)}

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

def init_weights_he(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.constant_(m.bias, 0)

class bde(torch.nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net
        
        # Get theta_k_star
        self.net.apply(init_weights_he)
        self.l_layer_params(self.net)

        # Get theta_k
        self.net.apply(init_weights_he)
        self.params = {k: v.clone().detach() for k, v in self.net.named_parameters()}

        self.device = next(self.net.parameters()).device

    def l_layer_params(self, net):
        theta_star_k = []
        sd = net.state_dict()
        sdk = sd.keys()
        for i,p in enumerate(sdk):
            if i < len(sdk) - 2:
                theta_star_k.append(sd[p].flatten(0))
            else:
                theta_star_k.append(torch.zeros(sd[p].flatten(0).shape))
        theta_star_k = torch.cat(theta_star_k).clone().detach()
        self.theta_star_k = theta_star_k

    def fnet(self, params, x):
        return functional_call(self.net, params, x)

    def jvp_func(self, theta_s,params,x):
        dparams = _dub(unflatten_like(theta_s, params.values()),params)
        _, proj = torch.func.jvp(lambda param: self.fnet(param, x),
                                (params,), (dparams,))
        proj = proj.detach()
        return proj    
    
    def forward(self, x):
        return self.net(x.to(self.device)) + self.jvp_func(self.theta_star_k, self.params, x.to(self.device))
    
def training_classification(net, trainloader, testloader, epochs, optimizer, scheduler, num_classes, verbose):
    # if verbose:
    #     pbar = tqdm.trange(epochs)
    # else:
    pbar = range(epochs)

    for epoch in pbar:
        train_loss, train_acc = train_loop(trainloader, net, torch.nn.MSELoss(), optimizer, scheduler, num_classes)
        test_loss, test_acc = test_loop(testloader, net, torch.nn.MSELoss(), num_classes)
        if verbose:
            # if epoch % 1 == 0:
            #     print("Epoch {} of {}".format(epoch,epochs))
            #     print("Training loss = {:.4f}".format(train_loss))
            #     print("Train accuracy = {:.1f}%".format(100*train_acc))
            #     print("Test loss = {:.4f}".format(test_loss))
            #     print("Test accuracy = {:.1f}%".format(100*test_acc))
            #     print("\n -------------------------------------")

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Loss: %.8f Acc@1: %.3f%% Test Loss: %.8f Test Acc@1: %.3f%%'
                    %(epoch, epochs, train_loss, train_acc*100, test_loss, test_acc*100))
            sys.stdout.flush()
    if verbose:
        print("Done!")
        print("Final training loss = {:.4f}".format(train_loss))
        print("Final train accuracy = {:.1f}%".format(100*train_acc))
        print("Final test loss = {:.4f}".format(test_loss))
        print("Final test accuracy = {:.1f}%".format(100*test_acc))
    return train_loss, train_acc, test_loss, test_acc

def train_loop(dataloader, net, loss_fn, optimizer, scheduler, num_classes):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x,y = x.to(net.device), y.to(net.device)
        # Compute prediction and loss
        pred = net(x)
        loss = loss_fn(torch.nn.functional.softmax(pred, -1), torch.nn.functional.one_hot(y, num_classes=num_classes).to(dtype=torch.float64))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    correct /= size

    return train_loss, correct


def test_loop(dataloader, net, loss_fn, num_classes):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y in dataloader:
            x,y = x.to(net.device), y.to(net.device)
            pred = net(x)
            test_loss += loss_fn(torch.nn.functional.softmax(pred, -1), torch.nn.functional.one_hot(y, num_classes=num_classes).to(dtype=torch.float64)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return test_loss, correct



    



