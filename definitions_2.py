import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from posteriors.lla.likelihoods import Categorical
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import posteriors.util as df

import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_dtype(torch.float64)

def train(dataloader, model, loss_fn, optimizer, scheduler = None, train_mode = True, regression = True):
    if train_mode:
        model.train()
    else:
        model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0
    res = {}

    for _, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        if not regression:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    res['loss'] = train_loss
    if not regression:
        correct /= size
        res['acc'] = correct
    return res
    
def test(dataloader, model, loss_fn, regression = True):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    res = {}

    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if not regression:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    res['loss'] = test_loss
    if not regression:
        correct /= size
        res['acc'] = correct
    return res
    
def training_2(train_loader, test_loader, model, loss_fn, epochs, optimizer, scheduler = None, regression = True, verbose = False, progress_bar = True, train_mode = True):
    '''
    Outputs:
        - train_loss
        - train_acc
        - test_loss
        - test_acc
    '''
    
    if progress_bar:
        pbar = tqdm.trange(epochs)
    else:
        pbar = range(epochs)

    for epoch in pbar:
        train_res = train(train_loader, model, loss_fn, optimizer, scheduler, train_mode=train_mode, regression=regression)
        test_res = test(test_loader, model, loss_fn, regression=regression)
        if verbose:
            if not progress_bar:
                print(f"Epoch {epoch} of {epochs}")
            print(f"train loss: {train_res['loss']:.4f}; test loss: {test_res['loss']:.4f}")
            if not regression:
                print(f"train acc: {train_res['acc']:.1%}; test acc: {test_res['acc']:.1%}")
            print()
    if not regression:
        return train_res['loss'], train_res['acc'], test_res['loss'], test_res['ac']
    else:
        return train_res['loss'], test_res['loss']


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, train_mode=True):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if train_mode:
        model.train()
    else:
        model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

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


def test_loop(dataloader, model, loss_fn, verbose=False):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return test_loss, correct

def training(train_loader, test_loader, model, loss_fn, optimizer, scheduler = None, epochs: int = 50, verbose: bool = False, progress_bar = True, train_mode = True):
    '''
    Training function. Will train and test, and will report metrics.

    Outputs:
        - train_loss
        - train_acc
        - test_loss
        - test_acc
    '''
    
    if progress_bar:
        pbar = tqdm.trange(epochs)
    else:
        pbar = range(epochs)

    for epoch in pbar:
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, scheduler, train_mode=train_mode)
        test_loss, test_acc = test_loop(test_loader, model, loss_fn)
        if verbose:
            if epoch % 1 == 0:
                print("Epoch {} of {}".format(epoch,epochs))
                print("Training loss = {:.4f}".format(train_loss))
                print("Train accuracy = {:.1f}%".format(100*train_acc))
                print("Test loss = {:.4f}".format(test_loss))
                print("Test accuracy = {:.1f}%".format(100*test_acc))
                print("\n -------------------------------------")
    if verbose:
        print("Done!")
        print("Final training loss = {:.4f}".format(train_loss))
        print("Final train accuracy = {:.1f}%".format(100*train_acc))
        print("Final test loss = {:.4f}".format(test_loss))
        print("Final test accuracy = {:.1f}%".format(100*test_acc))
    return train_loss, train_acc, test_loss, test_acc

def test_sampler(model,test,bs,probit=False):
    predictions = []
    test_dataloader = DataLoader(test,bs)
    for x,y in test_dataloader:
        predictions.append(model(x.to(device)).detach())
    predictions = torch.cat(predictions)
    if probit:
        return predictions.softmax(dim=1)
    else:
        return predictions


def ensemble_sampler_r(dataset,M,models,n_output,bs):
        ensemble_pred = torch.empty((M,len(dataset),n_output),device='cpu')
        for i in range(M):
            pred_i = df.evaluate_batch(dataset=dataset,model=models[i],batch_size=bs)
            ensemble_pred[i,:,:] = pred_i
        ensemble_pred_avg = ensemble_pred.mean(axis=0)
        ensemble_pred_var = ensemble_pred.var(axis=0)
        return ensemble_pred_avg, ensemble_pred_var

def ensemble_sampler(dataset,M,models,n_output,bs):
        ensemble_pred = torch.empty((M,len(dataset),n_output),device='cpu')
        for i in range(M):
            pred_i = df.evaluate_batch(dataset=dataset,model=models[i],batch_size=bs)
            ensemble_pred[i,:,:] = pred_i.detach()
        de_f = ensemble_pred.softmax(dim=2).mean(0)
        de_var = ensemble_pred.softmax(dim=2).var(0)
        return de_f, de_var, ensemble_pred.softmax(dim=2)

def swag_sampler(dataset,model,T,n_output,bs):
    swag_pred_samples = torch.empty((T,len(dataset),n_output),device='cpu')
    for t in range(T):
        model.sample(cov=True)
        pred_i = df.evaluate_batch(dataset=dataset, model=model, batch_size=bs)
        swag_pred_samples[t] = pred_i
    swag_probs = swag_pred_samples.softmax(dim=2).mean(axis=0)
    swag_var = swag_pred_samples.softmax(dim=2).var(axis=0)
    return swag_probs, swag_var, swag_pred_samples.softmax(dim=2)

def dropout_active(m):
    if isinstance(m,nn.Dropout):
        m.train()

def dropout_sampler(dataset, model, T, n_output, bs, logits=False):
    model.eval()
    model.apply(dropout_active)
    dropout_pred_samples = torch.empty((T,len(dataset),n_output),device='cpu')
    for t in range(T):
        mc_pred = df.evaluate_batch(dataset=dataset,model=model,batch_size=bs)
        dropout_pred_samples[t,:,:] = mc_pred
    if logits:
        return dropout_pred_samples.mean(axis=0), dropout_pred_samples.var(axis=0), dropout_pred_samples
    else:
        return dropout_pred_samples.softmax(dim=2).mean(axis=0), dropout_pred_samples.softmax(dim=2).var(axis=0), dropout_pred_samples.softmax(dim=2)
    
def lla_sampler(dataset, model, bs):
    lla_pred_samples = df.evaluate_batch_T(dataset=dataset,model=model,batch_size=bs)
    lla_pred_samples = lla_pred_samples.cpu()
    return lla_pred_samples.mean(axis=0), lla_pred_samples.var(axis=0), lla_pred_samples

def aucroc(id_scores,ood_scores):
    '''
    INPUTS: scores should be maximum softmax probability for each test example
    '''
    labels = np.zeros((id_scores.shape[0] + ood_scores.shape[0]), dtype=np.int32)
    labels[:id_scores.shape[0]] += 1
    examples = np.squeeze(np.hstack((id_scores, ood_scores)))
    return sk.roc_auc_score(labels, examples)

def ood_auc(id_scores,ood_scores):
    '''
    INPUTS: scores should be entropy for each test example
    '''
    labels = np.zeros((id_scores.shape[0] + ood_scores.shape[0]), dtype=np.int32)
    labels[id_scores.shape[0]:] += 1
    examples = np.squeeze(np.hstack((id_scores, ood_scores)))
    return sk.roc_auc_score(labels, examples)

def auc_var(id_var,ood_var):
    auc_roc_output = ood_auc(id_var.cpu().numpy(), ood_var.cpu().numpy())
    return auc_roc_output

def auc_metric(id_test, ood_test, logits=False):
    if logits:
        id_test = id_test.softmax(dim=1)
        ood_test = ood_test.softmax(dim=1)

    id_test, ood_test = id_test.detach(), ood_test.detach()

    id_dist = Categorical(probs=id_test)
    id_entropy = id_dist.entropy().cpu()

    ood_dist = Categorical(probs=ood_test)
    ood_entropy = ood_dist.entropy().cpu()

    ## OOD-AUC
    ood_auc_output = ood_auc(id_entropy,ood_entropy)

    ## AUCROC
    auc_roc_output = aucroc(id_test.cpu().numpy().max(1), ood_test.cpu().numpy().max(1))

    return ood_auc_output, auc_roc_output

def plt_vh(var_dict, nbin=20, title=None, probit_sum = True, show=False, save_file=None):
    l = len(var_dict.keys())
    if l <= 2:
        r, c = 1, 2
    elif l <= 4:
        r, c = 2, 2
    elif l <= 6:
        r, c = 2, 3
    elif l <= 8:
        r, c = 3, 3
                        
    f, ax = plt.subplots(r,c)
    f.set_figwidth(16)
    f.set_figheight(6)
    f.subplots_adjust(wspace=0.25,hspace=0.5)

    for i,key in enumerate(var_dict.keys()):
        if probit_sum:
            pi = array_to_numpy(var_dict[key]['id'].sum(1))
            po = array_to_numpy(var_dict[key]['ood'].sum(1))
        else:
            pi = array_to_numpy(var_dict[key]['id'])
            po = array_to_numpy(var_dict[key]['ood'])
        ax.reshape(-1)[i].hist(pi, bins=nbin, density=True, stacked=True, color='blue', alpha=1, label = 'ID')
        ax.reshape(-1)[i].hist(po, bins=nbin, density=True, stacked=True, color='red', alpha = 0.5, label = 'OOD')
        ax.reshape(-1)[i].set_title(key)
        ax.reshape(-1)[i].set_xlabel('$\sum_c \sigma_{c,i}^2$')
        ax.reshape(-1)[i].set_ylabel('Density')
        ax.reshape(-1)[i].legend()

    if len(var_dict.keys()) < len(ax.reshape(-1)): 
        for i in range(len(ax.reshape(-1)) - len(var_dict.keys())):
            ax.reshape(-1)[-(i+1)].remove()

    if title is not None:
        f.suptitle(title)
    
    plt.plot()

    if save_file is not None:
        plt.savefig(fname=save_file,format='pdf')
        
    if show:
        plt.show()

    plt.close()

def var_diff(var_dict):
    for key in var_dict.keys():
        print(f"\n---{key}---")
        id_var = var_dict[key]['id'].sum(1).mean(0)
        ood_var = var_dict[key]['ood'].sum(1).mean(0)
        print(f"mean sum of id var = {id_var:.4}")
        print(f"mean sum of ood var = {ood_var:.4}")
        print("relative diff of mean sums  = {:.5f}".format((ood_var - id_var)/id_var))

def probit_uncertainty(id_prob, ood_prob, ratio=True):
    '''
    INPUT: (N x C) probabilities, where N is number of test points, C is number of classes.
    '''
    pi = id_prob.sum(1).mean(0)
    po = ood_prob.sum(1).mean(0)
    if ratio:
        return po / pi
    else:
        return (po - pi) / pi
    
def dirichlet_uncertainty(id_prob, ood_prob, ratio=True):
    '''
    INPUT: (N) probabilities, where N is number of test points.
    '''
    pi = id_prob.mean(0)
    po = ood_prob.mean(0)
    if ratio:
        return po / pi
    else:
        return (po - pi) / pi

def array_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return 'ERR'

def violin_var(var_dict,probit_sum=True, title=None, save_file=None, show=False, text=False):
    plt.close()

    plt.figure(0,(9,6))
    plt.tight_layout()

    ticks = list(var_dict.keys())
    
    id_correct_var_list = []
    id_incorrect_var_list = []
    ood_var_list = []
    for key in var_dict.keys():
        pi_correct = array_to_numpy(var_dict[key]['id_correct'])
        pi_incorrect = array_to_numpy(var_dict[key]['id_incorrect'])
        po = array_to_numpy(var_dict[key]['ood'])
        if probit_sum:
            pi_correct = pi_correct.sum(1)
            pi_incorrect = pi_incorrect.sum(1)
            po = po.sum(1)
        id_correct_var_list.append(pi_correct.tolist())
        id_incorrect_var_list.append(pi_incorrect.tolist())
        ood_var_list.append(po.tolist())

    id_vars_plot = plt.violinplot(id_correct_var_list,
                                positions=np.array(
        np.arange(len(id_correct_var_list)))*2-0.6, 
                                points=100, widths=0.6,
                     showmeans=False, showextrema=False, showmedians=True)
    id_vars_plot_2 = plt.violinplot(id_incorrect_var_list,
                                positions=np.array(
        np.arange(len(id_incorrect_var_list)))*2, 
                                points=100, widths=0.6,
                     showmeans=False, showextrema=False, showmedians=True)
    ood_vars_plot = plt.violinplot(ood_var_list,
                                positions=np.array(
        np.arange(len(ood_var_list)))*2+0.6,
                                points=100, widths=0.6,
                     showmeans=False, showextrema=False, showmedians=True)
    

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
            
        # use plot function to draw a small line to name the legend.
        plt.plot([], c=color_code, label=label)
        plt.legend()
    
    # setting colors for each groups
    define_box_properties(id_vars_plot, '#008000', 'ID Correct')
    define_box_properties(id_vars_plot_2, '#D7191C', 'ID Incorrect')
    define_box_properties(ood_vars_plot, '#2C7BB6', 'OOD')
    
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel('$\sigma^2$')
    
    if title is not None:
        plt.title(title)

    plt.plot()

    if text:
        text_str = ''
        for m in var_dict.keys():
            if probit_sum:
                ood_av = var_dict[m]['ood'].sum(1).mean(0)
                id_correct_av = var_dict[m]['id_correct'].sum(1).mean(0)
            else:
                ood_av = var_dict[m]['ood'].mean(0)
                id_correct_av = var_dict[m]['id_correct'].mean(0)
            text_str += f' - {m}: R = {ood_av / id_correct_av:.2f}\n'
        props = dict(boxstyle='round', facecolor='#F5F5DC', alpha=0.5)
        plt.text(5.75, 0.2, text_str, fontsize=8, 
            verticalalignment='center', bbox=props)

    if save_file is not None:
        plt.savefig(fname=save_file,format='pdf')
        
    if show:
        plt.show()

    plt.close()

def sort_preds(pi,yi):
    index = (pi.mean(0).argmax(1) == yi)
    return pi[:,index,:], pi[:,~index,:]

def sort_dict_var(pred_dict,test_data):
    sorted_var_dict = {}
    for m in pred_dict.keys():
        pi_correct, pi_incorrect = sort_preds(pred_dict[m]['id'],test_data.targets)
        sorted_var_dict[m] = {'id_correct': pi_correct.var(0),
                            'id_incorrect': pi_incorrect.var(0),
                            'ood': pred_dict[m]['ood'].var(0)}
    return sorted_var_dict