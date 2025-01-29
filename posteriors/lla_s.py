from torch.func import vmap, jacrev, jvp
from functorch import make_functional, make_functional_with_buffers
from torch.utils.data import DataLoader
import torch
import tqdm
import numpy as np
import copy
torch.set_default_dtype(torch.float64)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# def classification(net,train,test,S,epochs,lr,epochs_mode,lr_mode,alpha0,alpha_it=5,beta=1,regression=True,scheduler_true=False,verbose=False,progress_bar=True):
#     ## Create loaders and get entire training set
#     train_loader_total = DataLoader(train,batch_size=len(train))
#     X_train,_ = next(iter(train_loader_total))

#     ## Compute jacobian of net, evaluated on training set
#     fnet, params = make_functional(net)
#     def fnet_single(params, x):
#         return fnet(params, x.unsqueeze(0)).squeeze(0)
#     J = vmap(jacrev(fnet_single), (None, 0))(params, X_train.to(device))
#     J = [j.detach().flatten(1) for j in J]
#     J = torch.cat(J,dim=1).detach()
#     n, p = J.shape

#     # Set progress bar
#     if progress_bar:
#         pbar = tqdm.trange(alpha_it)
#     else:
#         pbar = range(alpha_it)

#     # Get MAP parameters
#     theta_t = flatten(params)
#     pp = copy.deepcopy(params)

#     # Create empty tensors to store posterior samples
#     zeta = torch.empty((p,S), requires_grad=False)
#     zeta_exact = torch.empty((p,S), requires_grad=False)

#     # Create dict for results
#     res = {'sample_loss': [],
#            'mode_loss': []}

#     # Set initial hyperparameters
#     alpha = alpha0

#     # # Find g-prior and scale Embedding (Jacobian) by it
#     # thetaprime = 1/alpha * J.T @ torch.randn((n,S), requires_grad=False) * (beta)**(-0.5) # n x s
#     # s = 1/alpha * (1/S * torch.sum(torch.square(thetaprime),dim=1))**(-0.5)

#     # J = J @ torch.diag(s)

#     # Create new theta_star_0
#     theta_star = []
#     for pi in pp:
#         theta_star.append(torch.nn.parameter.Parameter(torch.zeros(size=pi.shape,device=device)))
#     theta_star = tuple(theta_star)

#     for i in pbar:
#         print(f'Alpha = {alpha}')

#         # Exact GD

#         # Random variables
#         Eps = torch.randn((n,S), requires_grad=False) * (beta)**(-0.5)
#         theta0 = torch.randn((p,S), requires_grad=False) * (alpha)**(-0.5)

#         # Create new parameters
#         z = theta0.detach().clone()

#         # Create loss function
#         L = lambda z, s : 0.5 * beta * (z[:,s].T @ (J.T @ (J @ z[:,s]))) + 0.5 * alpha * (z[:,s] - theta0[:,s] - beta/alpha * J.T @ Eps[:,s]).T @ (z[:,s] - theta0[:,s] - beta/alpha * J.T @ Eps[:,s])

#         for i in range(epochs):
#             g = beta * J.T @ (J @ z - Eps) + alpha * (z - theta0)
#             z -= lr * g
#         loss = 0
#         for s in range(S):
#             loss += L(z,s)
#         res['sample_loss'].append(loss / S)


#         ## --- Get linearised mode --- ##

#         # Create linearized network function
#         f_lin = linearize(fnet, params)

#         # Training    
#         optim = torch.optim.SGD(theta_star,lr=lr,momentum=0.9,weight_decay=alpha)
#         if not regression:
#             loss_fn = torch.nn.CrossEntropyLoss()
#         else:
#             loss_fn = torch.nn.MSELoss()
#         if scheduler_true:
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs)

#         train_loader = DataLoader(train, 50)

#         for epoch in range(epochs):
#             loss_e = 0
#             acc_e = 0
#             for x,y in train_loader:
#                 x, y = x.to(device), y.to(device)
#                 optim.zero_grad()
#                 pred = f_lin(theta_star, x)
#                 loss = loss_fn(pred, y)
#                 loss.backward()
#                 loss_e += loss.item()
#                 if not regression:
#                     acc_e += (pred.argmax(1) == y).type(torch.float).sum().item()
#                 optim.step()
#             if scheduler_true:
#                 scheduler.step()
            
#             loss_e /= len(train_loader)
#             if not regression:
#                 acc_e /= len(train_loader.dataset)
#             if verbose:
#                 print(f"\nepoch: {epoch}")
#                 print(f"loss: {loss_e}")
#                 if not regression:
#                     print(f"acc: {acc_e:.1%}")
#         res['mode_loss'].append(loss_e)

#         ## --- Select new hyperparameters --- ##
#         gamma_hat = 0
#         for s in range(S):
#             gamma_hat += z[:,s].T @ J.T @ J @ z[:,s]
#         gamma_hat *= beta/S

#         alpha = (gamma_hat / (torch.linalg.norm(flatten(theta_star))**2)).item()

def small_regression_parallel(net,train,test,S,epochs,lr,epochs_mode,lr_mode,alpha0,alpha_it=5,beta=1,exact=False,regression=True,scheduler_true=False,verbose=False,progress_bar=True):
    ## Create loaders and get entire training set
    train_loader_total = DataLoader(train,batch_size=len(train))
    X_train,_ = next(iter(train_loader_total))

    ## Compute jacobian of net, evaluated on training set
    fnet, params = make_functional(net)
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    J = vmap(jacrev(fnet_single), (None, 0))(params, X_train.to(device))
    J = [j.detach().flatten(1) for j in J]
    J = torch.cat(J,dim=1).detach()
    n, p = J.shape

    # Set progress bar
    if progress_bar:
        pbar = tqdm.trange(alpha_it)
    else:
        pbar = range(alpha_it)

    # Get MAP parameters
    theta_t = flatten(params)
    pp = copy.deepcopy(params)

    # Create empty tensors to store posterior samples
    zeta = torch.empty((p,S), requires_grad=False)
    zeta_exact = torch.empty((p,S), requires_grad=False)

    # Create dict for results
    res = {'sample_loss': [],
           'mode_loss': []}

    # Set initial hyperparameters
    alpha = alpha0

    # # Find g-prior and scale Embedding (Jacobian) by it
    # thetaprime = 1/alpha * J.T @ torch.randn((n,S), requires_grad=False) * (beta)**(-0.5) # n x s
    # s = 1/alpha * (1/S * torch.sum(torch.square(thetaprime),dim=1))**(-0.5)

    # J = J @ torch.diag(s)

    # Create new theta_star_0
    theta_star = []
    for pi in pp:
        theta_star.append(torch.nn.parameter.Parameter(torch.zeros(size=pi.shape,device=device)))
    theta_star = tuple(theta_star)

    for i in pbar:
        print(f'Alpha = {alpha}')

        # Exact GD

        # Random variables
        Eps = torch.randn((n,S), requires_grad=False) * (beta)**(-0.5)
        theta0 = torch.randn((p,S), requires_grad=False) * (alpha)**(-0.5)

        # Create new parameters
        z = theta0.detach().clone()

        # Exact
        if exact:
            with torch.no_grad():
                z = torch.linalg.solve(beta * J.T @ J + alpha * torch.eye(p), alpha * theta0 + beta * J.T @ Eps).detach()
        else:
            # Create loss function
            L = lambda z, s : 0.5 * beta * (z[:,s].T @ (J.T @ (J @ z[:,s]))) + 0.5 * alpha * (z[:,s] - theta0[:,s] - beta/alpha * J.T @ Eps[:,s]).T @ (z[:,s] - theta0[:,s] - beta/alpha * J.T @ Eps[:,s])

            for i in range(epochs):
                g = beta * J.T @ (J @ z - Eps) + alpha * (z - theta0)
                z -= lr * g
            loss = 0
            for s in range(S):
                loss += L(z,s)
            res['sample_loss'].append(loss / S)

        


        ## --- Get linearised mode --- ##

        # Create linearized network function
        f_lin = linearize(fnet, params)

        # Training    
        optim = torch.optim.SGD(theta_star,lr=lr,momentum=0.9,weight_decay=alpha)
        if not regression:
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()
        if scheduler_true:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs)

        train_loader = DataLoader(train, 50)

        for epoch in range(epochs):
            loss_e = 0
            acc_e = 0
            for x,y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                pred = f_lin(theta_star, x)
                loss = loss_fn(pred, y)
                loss.backward()
                loss_e += loss.item()
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
        res['mode_loss'].append(loss_e)

        ## --- Select new hyperparameters --- ##
        gamma_hat = 0
        for s in range(S):
            gamma_hat += z[:,s].T @ J.T @ J @ z[:,s]
        gamma_hat *= beta/S

        alpha = (gamma_hat / (torch.linalg.norm(flatten(theta_star))**2)).item()

    ## Create loaders and get entire training set
    test_loader_total = DataLoader(test,batch_size=len(test))
    X_test,_ = next(iter(test_loader_total))

    ## Compute jacobian of net, evaluated on test set
    J = vmap(jacrev(fnet_single), (None, 0))(params, X_test.to(device))
    J = [j.detach().flatten(1) for j in J]
    J = torch.cat(J,dim=1).detach()
    n, p = J.shape

    with torch.no_grad():
        pred_s = []
        f_nlin = net(X_test)
        for s in range(S):
            pred_lin = J @ (z[:,s] - theta_t) + f_nlin.squeeze(1)
            pred_s.append(pred_lin)

    pred_lla = torch.stack(pred_s)
    return pred_lla, res

def regression_small(train, net, a_it=5, beta=1.0, alpha=1.0):
    train_loader_total = torch.utils.data.DataLoader(train,batch_size=len(train))
    X_train,Y_train = next(iter(train_loader_total))

    # Initial parameters
    # alpha = 1.0
    k = 1000

    # Method parameters
    a_tol = 1e-2

    epochs_posterior = 1500
    lr_posterior = 1e-2

    epochs_mode = 1500
    lr_mode = 1e-2
    mu_mode = 0.9

    # Get Jacobian Design Matrix Phi
    fnet, params = make_functional(net)
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    J = vmap(jacrev(fnet_single), (None, 0))(params, X_train.to(device))
    J = [j.detach().flatten(1) for j in J]
    J = torch.cat(J,dim=1).detach()
    n, p = J.shape

    # Vector of MAP parameters

    theta_t = flatten(params).unsqueeze(1)

    # Initialize regularisers
    theta0 = torch.empty((p,k))
    thetaprime = torch.empty((p,k))
    zeta = torch.empty((p,k))
    theta_hat = torch.zeros((p,1))

    # Sample initial regulariser/posterior samples
    for j in range(k):
        theta0[:,j] = (alpha**(-0.5))*torch.randn((p), requires_grad=False)
        thetaprime[:,j] = (beta**0.5)/alpha * J.to(device).T @ torch.randn((n),requires_grad=False, device=device)
        zeta[:,j] = theta0[:,j]
    s = (1/(alpha*k) * torch.sum(torch.square(thetaprime),dim=1)**(-0.5)).to(device)

    ## Posterior samples
    a_i = 0
    # while a_i < a_it:
        # Sample from posterior
    for j in range(k):
        # Copy regularisers
        theta0_j = theta0[:,j].clone().detach().to(device)
        thetaprime_j = thetaprime[:,j].clone().detach().to(device)
        zeta_j = zeta[:,j].clone().detach().to(device)

        # # GD scheme
        # l = lambda z : beta * (s * z).unsqueeze(1).T @ J.to(device).T @ J.to(device) @ (s * z) + alpha * torch.linalg.norm(z - theta0_j - (s * thetaprime_j))*2
        # z = torch.nn.parameter.Parameter(zeta_j)
        # optim = torch.optim.SGD([z],lr=lr_posterior, momentum=0.9)
        # for i in range(epochs_posterior):
        #     loss = l(z)
        #     loss.backward()
        #     optim.step()
        #     optim.zero_grad()
        # zeta[:,j] = z.detach()

        optimal_zeta = torch.linalg.solve(alpha*torch.eye(p) + beta * J.T @ J, alpha * theta0_j + alpha * thetaprime_j)
        # print(f'sgd vs exact : {torch.linalg.norm(z-optimal_zeta)}')
        zeta[:,j] = optimal_zeta.detach()

            # print(f'---- parameter samples ----')
            # print(f'sample {j}, loss = {loss.item():.4f}')

        # # Find posterior mode
        # theta = theta_hat.clone().detach().to(device)
        # for epoch in range(epochs_mode):
        #     f_nlin = net(X_train.to(device))
        #     f_lin = (J.to(device) @ (theta - theta_t) + f_nlin).detach()
        #     resid = f_lin - Y_train.to(device)
        #     grad = J.T.to(device) @ resid.to(device) / X_train.shape[0] + alpha * theta

        #     if epoch == 0:
        #         bt = grad
        #     else:
        #         bt = mu_mode*bt + grad

        #     theta -= lr_mode * bt

        #     loss = torch.mean(torch.square(J @ (theta - theta_t) + f_nlin - Y_train.to(device))).max()

        #     # if epoch % 10 == 0:
        #     #     print("\n-----------------")
        #     #     print("Epoch {} of {}".format(epoch,epochs_mode))
        #     #     print("max l2 loss = {}".format(torch.mean(torch.square(J @ (theta - theta_t) + f_nlin - Y_train.to(device))).max()))
        #     #     print("Residual of normal equation l2 = {}".format(torch.mean(torch.square(J.T @ ( f_nlin + J @ (theta - theta_t) - Y_train.to(device))))))

        # # Calculate gamma
        # gamma = 0
        # for j in range(k):
        #     gamma += ((J @ zeta[:,j]).T @ J @ zeta[:,j]).item()
        # gamma *= beta/k

        # # gamma = beta/k * torch.sum(torch.tensor([zeta[:,j].T @ J.T @ J @ zeta[:,j] for j in range(k)]))
        # # gamma = (beta**0.5)/k * torch.sum(torch.square((zeta.to(device) * s.unsqueeze(1)).T @ J.T),(0,1))

        # # Calculate alpha'
        # alpha_prime = (gamma / torch.norm(theta)**2).cpu()

        # # Update parameters
        # theta0 *= torch.sqrt(alpha/alpha_prime)
        # thetaprime *= (alpha/alpha_prime)

        # alpha_prev = alpha
        # alpha = alpha_prime
        # alpha_diff = torch.norm(alpha - alpha_prev)
        # print(f'\n------ iteration {a_i} ------')
        # print(f'alpha = {alpha:.3f}, alpha_diff = {alpha_diff:.4e}')
        # # if alpha_diff < a_tol:
        # #     break
        # a_i += 1

    return zeta

def test_preds(zeta, test, net):
    test_loader = torch.utils.data.DataLoader(test,batch_size=len(test))

    fnet, params = make_functional(net)
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    theta_t = flatten(params)

    print(zeta.get_device())
    print(theta_t.get_device())

    # Concatenate predictions
    pred_s = []
    for x,y in test_loader:
        x, y = x.to(device), y.to(device)
        f_nlin = net(x)
        J = vmap(jacrev(fnet_single), (None, 0))(params, x)
        J = [j.detach().flatten(1) for j in J]
        J = torch.cat(J,dim=1)

        pred_lin = J.to(device) @ (zeta.to(device) - theta_t.unsqueeze(1)) + f_nlin ## n x S
        pred_s.append(pred_lin.detach().cpu())

    predictions = torch.cat(pred_s,dim=0)
    return predictions

def classification(net, train, train_bs, k, n_output, alpha0, alpha_it, post_lr, post_epochs, lin_lr, lin_epochs, compute_exact=False):
    def VJP(v,params,x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        _, vjp_fn = torch.func.vjp(lambda param: fnet(param, x), params)
        if len(v.shape) < 2:
            v = v.unsqueeze(0)
        vjp = vjp_fn(v)
        return vjp

    def jvp_s(zeta_j,params,x):
        v = tuple(unflatten_like(zeta_j, params))
        _, proj = torch.func.jvp(lambda param: fnet(param, x),
                                (params,), (v,))
        return proj

    
    fnet, params = make_functional(net)
    f_lin = linearize(fnet, params)

    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    train_loader = DataLoader(train,batch_size=len(train))
    
    J = vmap(jacrev(fnet_single), (None, 0))(params, train_x.to(device))
    J = [j.detach().flatten(2) for j in J]
    J = torch.cat(J,dim=2).flatten(0,1)

    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    alpha = alpha0
    mu = 0.9

    

    ### --- Initial Samples --- ###
    theta0 = torch.empty((p,k))
    thetaprime = torch.empty((p,k), device=device)
    inv_scale_vec = torch.zeros((p), device=device) # Add squared elements of thetaprime for each batch in each realisation
    train_loader = DataLoader(train, batch_size=1)

    print('Computing initial samples')
    for j in range(k):

        theta0[:,j] = torch.randn((p), requires_grad=False) * (alpha)**(-0.5)
        
        thetaprimej = torch.zeros((p), device=device)

        for x,_ in train_loader:
            bi = Bi(net,x.to(device)).squeeze(0)
            U,S,_ = torch.linalg.svd(bi)
            S_inv = 1 / S
            eps = torch.randn((bi.shape[0]))
            chol_eps = U @ ((S**0.5) * eps.to(device))
            chol_inv_eps = U @ ((S_inv**0.5) * eps.to(device))
            J_chol_eps = VJP(chol_eps.unsqueeze(0),params,x.to(device))
            J_chol_eps = [j.detach().flatten(0) for j in J_chol_eps[0]]
            J_chol_eps = torch.cat(J_chol_eps,dim=0).detach().T
            thetaprimej += J_chol_eps

            inv_scale_vec += torch.square(J_chol_eps)
        thetaprimej *= (alpha**-1)

        thetaprime[:,j] = thetaprimej

    s = torch.sqrt(k / inv_scale_vec)
    # s = torch.ones((p))

    # Set theta_mode to zero
    theta_mode = []
    for pi in params:
        theta_mode.append(torch.nn.parameter.Parameter(torch.zeros(size=pi.shape,device=device)))
    theta_mode = tuple(theta_mode)

    # for ai in range(alpha_it):

    train_loader = DataLoader(train, batch_size=train_bs)

    # Get samples (parallel) (batch)
    zeta = thetaprime.detach().clone() # p x k

    print(f'Finding posterior samples')
    for epoch in tqdm.trange(post_epochs):
        Gz_k = torch.zeros((p,k), device=device)
        for x,_ in train_loader:
            J = vmap(jacrev(fnet_single), (None, 0))(params, x.to(device))
            J = [j.detach().flatten(2) for j in J]
            J = torch.cat(J,dim=2).flatten(0,1) # NC x P

            # Jz = vmap(jvp_s, (1,None,None))(zeta*s.unsqueeze(1),params,x.to(device)).permute(1,2,0) # K x N x C--> N x C x K

            Jz = J @ (zeta*s.unsqueeze(1)) # NC x K

            H = torch.block_diag(*Bi(net,x.to(device))) # NC x NC
            # HJz = torch.bmm(H,Jz) # N x C x K
            HJz = H @ Jz # NC x K

            # JHJz_batch = lambda v : vmap(VJP,(0,None,0))(v, params, x.to(device)) # v : N x C x 1

            # JHJz_samples = vmap(JHJz_batch, (2))(HJz) 
            # JHJz = torch.cat([j.detach().flatten(2) for j in JHJz_samples[0]],dim=2) # K x N x P

            JHJz = (J.T * s.unsqueeze(1)) @ HJz # P x K

            # JHJz = torch.sum(JHJz,dim=0) # Sum over batch dimension (N)
            # JHJz *= (s.unsqueeze(0)) # Scale jacobian by s
            Gz_k += JHJz

        g = Gz_k + alpha*(zeta - theta0.to(device) - (s.unsqueeze(1) * thetaprime))

        l = 0.5 * torch.sum(zeta * Gz_k, 0) + 0.5 * alpha * torch.sum((zeta - theta0.to(device) - (s.unsqueeze(1) * thetaprime)) * 
                                                                (zeta - theta0.to(device) - (s.unsqueeze(1) * thetaprime)), dim = 0)
        
        zeta -= post_lr * g

        if epoch % 10 == 0:
            print(f'loss = {l}')
            print(f'grad ||.|| = {torch.linalg.norm(g):.3}')
            
        # if epoch == 0:
        #     bt = g
        # else:
        #     bt = mu*bt + g

    if compute_exact:

        print('Computing exact solution')
        # Compute exact solution for comparison
        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)
        J = jacrev(fnet_single)
        train_loader = DataLoader(train, batch_size=1)
        exact = torch.empty((p,k))

        G = torch.zeros((p,p))
        for x,_ in train_loader:
            Jx = J(params,x)
            Jx = torch.cat([j.detach().flatten(2) for j in Jx],dim=2).squeeze(0)
            scaled_Jx = Jx * s
            H = Bi(net,x).squeeze(0)
            G += scaled_Jx.T @ H @ scaled_Jx

        M = G + alpha * torch.eye(p)

        l = torch.empty((k))

        for j in range(k):
            v = (s*thetaprime[:,j]) + theta0[:,j]

            exact[:,j] = torch.linalg.solve(M,alpha*v)

        print(f'exact vs sto : diff = {torch.linalg.norm((zeta - exact), dim=0)}')

    # Get linear map
    f_lin = linearize(fnet, params, scale=s)

    optim = torch.optim.SGD(tuple(theta_mode),lr=lin_lr,momentum=mu,weight_decay=alpha)
    loss_fn = torch.nn.CrossEntropyLoss()

    print('Finding linear MAP')
    for epoch in tqdm.trange(lin_epochs):
        loss_e = 0
        acc_e = 0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = f_lin(theta_mode, x)
            loss = loss_fn(pred, y)
            loss.backward()
            loss_e += loss
            acc_e += (pred.argmax(1) == y).type(torch.float).sum().item()
            optim.step()
        
        loss_e /= len(train_loader)
        acc_e /= len(train_loader.dataset)
        print(f"\nepoch: {epoch}")
        print(f"loss: {loss_e}")
        print(f"acc: {acc_e:.1%}")

    # Update alpha
    eff_dim = 1 / k * torch.sum((zeta * Gz_k),dim=(0,1))
    new_alpha = (eff_dim / torch.linalg.norm(flatten(theta_mode))**2).detach().cpu().item()
    theta0 *= (alpha / new_alpha)**0.5
    thetaprime *= alpha / new_alpha
    alpha = new_alpha
    print(f'alpha = {alpha}')

    return zeta, s

def classification_prediction(net, zeta, test, test_bs, scale):
    fnet, params = make_functional(net)
    f_lin = linearize(fnet, params, scale=scale)
    test_loader = DataLoader(test,test_bs)

    # Zeta should have size (p x k)
    preds = []
    for j in range(zeta.shape[1]):
        pred_s = []
        p = unflatten_like(zeta[:,j],params)
        for x,_ in test_loader:
            pred_s.append(f_lin(p, x.to(device)))
        preds.append(torch.cat(pred_s,dim=0))
    preds = torch.stack(preds)
    
    return preds

def classification_small(net, train, train_bs, k, n_output, alpha0, alpha_it, post_lr, post_epochs, lin_lr, lin_epochs, compute_sgd=True, compute_exact=False):
    fnet, params = make_functional(net)
    f_lin = linearize(fnet, params)

    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    train_loader = DataLoader(train,batch_size=len(train))
    train_x,train_y = next(iter(train_loader))
    
    J = vmap(jacrev(fnet_single), (None, 0))(params, train_x.to(device))
    J = [j.detach().flatten(2) for j in J]
    J = torch.cat(J,dim=2)
    J_flat = J.flatten(0,1)

    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    alpha = alpha0
    mu = 0.9

    ### --- Initial Samples --- ###
    theta0 = torch.empty((p,k), requires_grad=False)
    thetaprime = torch.empty((p,k), device=device, requires_grad=False)
    inv_scale_vec = torch.zeros((p), device=device, requires_grad=False) # Add squared elements of thetaprime for each batch in each realisation
    train_loader = DataLoader(train, batch_size=1)
    Eps = torch.empty((J.shape[0]*J.shape[1],k))

    print('Computing initial samples')
    for j in range(k):

        theta0[:,j] = torch.randn((p), requires_grad=False) * (alpha)**(-0.5)
        
        thetaprimej = torch.zeros((p), device=device, requires_grad=False)
        epsj = []

        for i,(x,_) in enumerate(train_loader):
            bi = Bi(net,x.to(device)).squeeze(0)
            U,S,_ = torch.linalg.svd(bi)
            S_inv = 1 / S
            eps = torch.randn((bi.shape[0]))
            epsj.append(eps.reshape(-1))
            chol_eps = U @ ((S**0.5) * eps.to(device))
            chol_inv_eps = U @ ((S_inv**0.5) * eps.to(device))
            J_chol_eps = J[i,:,:].T @ chol_eps
            thetaprimej += J_chol_eps

        inv_scale_vec += torch.square(thetaprimej)
        thetaprimej *= alpha

        thetaprime[:,j] = thetaprimej
        Eps[:,j] = torch.cat(epsj)

    # s = torch.sqrt(k / inv_scale_vec)
    # s = torch.ones((p))

    print(f'theta0_cov = {torch.cov(theta0)}')
    print(f'theta_0_cov exact = {(alpha**-1)*torch.eye(p)}')

    print(f'thetaprime_cov = {torch.cov(thetaprime)}')
    print(f'thetaprime_cov exact = {(alpha**-1)*J_flat.T @ torch.block_diag(*Bi(net,train_x.to(device))) @ J_flat}')



    # Set theta_mode to zero
    theta_mode = []
    for pi in params:
        theta_mode.append(torch.nn.parameter.Parameter(torch.zeros(size=pi.shape,device=device)))
    theta_mode = tuple(theta_mode)

    for ai in range(alpha_it):
        print(f'\n\nalpha = {alpha}')

        # ----------- Compute posterior samples ----------- #
        if compute_exact:
            # Compute exact solution
            print('Computing exact solution')
            exact = torch.empty((p,k))

            G = (J_flat).T @ torch.block_diag(*Bi(net,train_x.to(device))) @ (J_flat)

            M = G + alpha * torch.eye(p, device=device)

            for j in range(k):
                v = thetaprime[:,j] + theta0[:,j]

                exact[:,j] = torch.linalg.solve(M,alpha*v)

        if compute_sgd:
            # Compute sgd solution
            train_loader = DataLoader(train, batch_size=len(train))

            def jacobian(x):
                J = vmap(jacrev(fnet_single), (None, 0))(params, x.to(device))
                J = [j.detach().flatten(2) for j in J]
                J = torch.cat(J,dim=2)
                return J.flatten(0,1)

            zeta = torch.empty((p,k))
            # Compute in series
            for j in range(k):
                print(f'Sample {j}')
                theta0j = theta0[:,j].detach().clone()
                thetaprimej = thetaprime[:,j].detach().clone()
                zetaj = thetaprimej.detach().clone() # p x k
                zetaj.requires_grad = True

                L = lambda z,x : z.T @ jacobian(x).T @ torch.block_diag(*Bi(net,x)).detach() @ jacobian(x) @ z + alpha * torch.dot((z - theta0j - thetaprimej),
                                                                                                                        (z - theta0j - thetaprimej))
                optimizer = torch.optim.SGD([zetaj], lr=post_lr, momentum=0.9)

                for epoch in range(post_epochs):
                    l = 0
                    for x,_ in train_loader:
                        # Compute loss
                        loss = L(zetaj,x.to(device))
                        l += loss.item()

                        # Backpropagation
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    l /= len(train_loader)
                print(f'finished, loss: {l:.4}')
                if compute_exact:
                    print(f'exact vs sto : diff = {torch.linalg.norm((zetaj - exact[:,j]), dim=0)}')
                
                zeta[:,j] = zetaj.detach()


        # Get linear map
        f_lin = linearize(fnet, params, scale=None)

        optim = torch.optim.SGD(tuple(theta_mode),lr=lin_lr,momentum=mu,weight_decay=alpha)
        loss_fn = torch.nn.CrossEntropyLoss()

        print('Finding linear MAP')
        for epoch in tqdm.trange(lin_epochs):
            loss_e = 0
            acc_e = 0
            for x,y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                pred = f_lin(theta_mode, x)
                loss = loss_fn(pred, y)
                loss.backward()
                if alpha > 100:
                    torch.nn.utils.clip_grad_value_(tuple(theta_mode), 1/p)
                loss_e += loss
                acc_e += (pred.argmax(1) == y).type(torch.float).sum().item()
                optim.step()
            
            loss_e /= len(train_loader)
            acc_e /= len(train_loader.dataset)

            # if epoch % 10 == 0:
            #     print(f"loss: {loss_e}")
            #     print(f"acc: {acc_e:.1%}")
        print(f"finished")
        print(f"loss: {loss_e}")
        print(f"acc: {acc_e:.1%}")

        # Exact dim
        M = J_flat.T @ torch.block_diag(*Bi(net,train_x.to(device))) @ J_flat
        exact_eff_dim = torch.linalg.solve(M + alpha*torch.eye(p), M)
        exact_eff_dim = torch.trace(exact_eff_dim)

        # Calculate effective dim
        eff_dim = 0
        for j in range(k):
            if compute_exact and not compute_sgd:
                eff_dim += exact[:,j].T @ M @ exact[:,j]
            else:
                eff_dim += zeta[:,j].T @ M @ zeta[:,j]
        eff_dim /= k
        eff_dim = torch.clamp(eff_dim,1,p-1)
        print(f'eff_dim = {eff_dim}')
        print(f'exact eff_dim = {exact_eff_dim}')
        print(f'theta_mode norm = {torch.linalg.norm(flatten(theta_mode))**2}')
        
        # Update alpha
        new_alpha = (exact_eff_dim / torch.linalg.norm(flatten(theta_mode))**2).detach().cpu().item()
        theta0 *= (alpha / new_alpha)**0.5
        thetaprime *= alpha / new_alpha
        alpha = new_alpha

        # Test convergence to theta0
        theta0_diff = torch.linalg.norm(exact - theta0)
        print(f'theta0 convergence = {theta0_diff:.4}')

    if compute_exact and not compute_sgd:
        return exact
    else:
        return zeta

def classification_prediction(net, zeta, test, test_bs, scale):
    fnet, params = make_functional(net)
    f_lin = linearize(fnet, params, scale=None)
    test_loader = DataLoader(test,test_bs)

    # Zeta should have size (p x k)
    preds_lla = []
    for j in range(zeta.shape[1]):
        pred_s = []
        p = unflatten_like(zeta[:,j],params)
        for x,_ in test_loader:
            pred_s.append(f_lin(p, x.to(device)))
        preds_lla.append(torch.cat(pred_s,dim=0))
    preds_lla = torch.stack(preds_lla)

    map_preds = []
    for x,_ in test_loader:
        map_preds.append(net(x.to(device)).detach())
    map_preds = torch.cat(map_preds)
    
    return map_preds, preds_lla



def Bi(net, x):
    pi = net(x)
    pi = torch.nn.functional.softmax(pi, dim=1)
    pi_diag = torch.diag_embed(pi)
    pi_ip = torch.bmm(pi.unsqueeze(2), pi.unsqueeze(2).permute(0,2,1))
    Bi_batch = pi_diag - pi_ip + 1e-6 * torch.eye(pi_diag.shape[-1]).unsqueeze(0).to(device)
    return Bi_batch

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def linearize(f, params, scale=None):
  def f_lin(p, *args, **kwargs):
    dparams = _sub(p, params)
    if scale is not None:
        dparams = _mul(unflatten_like(scale,dparams),dparams)
    f_params_x, proj = torch.func.jvp(lambda param: f(param, *args, **kwargs),
                           (params,), (dparams,))
    return proj #f_params_x + proj
  return f_lin

def _sub(x, y):
  return tuple(x - y for (x, y) in zip(x, y))

def _mul(x,y):
    return tuple(x * y for (x,y) in zip(x,y))

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