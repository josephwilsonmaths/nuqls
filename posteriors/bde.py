import torch
device = 'cpu'

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

def train_bde(train_loader,net,delta,theta_k,loss_fn,Lambda,optim,sched):
    train_loss = 0
    for i, (X,y) in enumerate(train_loader):
        X,y = X.to(device), y.to(device)
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