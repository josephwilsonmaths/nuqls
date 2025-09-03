import numpy as np
import sklearn.metrics as sk
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassCalibrationError
import utils.datasets as ds
import time

def compute_metrics(test_loader, test_preds, ood_test_preds, samples=True):
    # If samples = True -> Assume that test_preds is S x N x C, in probit space, and detached, on cpu.
    # If samples = False -> Assume that test_preds is N x C, in probit space, and detached, on cpu.
    if samples:
        num_classes=test_preds.shape[2]
        mean_prediction = test_preds.mean(0); ood_mean_predictions = ood_test_preds.mean(0)
    else:
        num_classes=test_preds.shape[1]
        mean_prediction = test_preds; ood_mean_predictions = ood_test_preds

    ece_compute = MulticlassCalibrationError(num_classes=num_classes,n_bins=10,norm='l1')
    loss_fn = torch.nn.NLLLoss(reduction='mean')
    test_targets = torch.cat([y for _,y in test_loader])

    ce = loss_fn(torch.log(mean_prediction),test_targets).item()
    acc = (mean_prediction.argmax(1) == test_targets).type(torch.float).mean().item()
    ece = ece_compute(mean_prediction, test_targets).cpu().item()
    ood_auc, auc_roc = auc_metric(mean_prediction, ood_mean_predictions, logits=False)

    if samples:
        var_roc = aucroc(ood_test_preds.var(0).sum(1), test_preds.var(0).sum(1))
        idc, idic = sort_preds(test_preds,test_targets)
        var_roc_id = aucroc(idic.var(0).sum(1), idc.var(0).sum(1))
        var_roc_ood = aucroc(ood_test_preds.var(0).sum(1), idc.var(0).sum(1))
        vmsp_dict = sort_probabilies(test_preds, ood_test_preds, test_data=test_loader.dataset)
        return ce, acc, ece, ood_auc, auc_roc, var_roc, var_roc_id, var_roc_ood, vmsp_dict
    else:
        return ce, acc, ece, ood_auc, auc_roc

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
    from posteriors.lla.likelihoods import Categorical
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

def sort_preds(pi,yi):
    index = (pi.mean(0).argmax(1) == yi)
    return pi[:,index,:], pi[:,~index,:]

def sort_preds_logit(pi,yi):
    index = (pi.softmax(2).mean(0).argmax(1) == yi)
    return pi[:,index,:], pi[:,~index,:]

def sort_probabilies(id_probs,ood_probs,test_data):
    '''
        id_probs and ood_probs should be shape S x N x C, where S is number of samples, N is number of datapoints, and C is number of classes.
    '''

    # Get targets
    full_test_loader = DataLoader(test_data,len(test_data))
    _,test_target = next(iter(full_test_loader))

    pi_correct, pi_incorrect = sort_preds(id_probs,test_target)
            
    max_index = pi_correct.mean(0).argmax(1)
    pi_correct_var = pi_correct[:,range(len(max_index)),max_index].var(0)
    
    max_index = pi_incorrect.mean(0).argmax(1)
    pi_incorrect_var = pi_incorrect[:,range(len(max_index)),max_index].var(0)

    max_index = ood_probs.mean(0).argmax(1)
    po_var = ood_probs[:,range(len(max_index)),max_index].var(0)

    prob_var_dict = {'id_correct': pi_correct_var,
                    'id_incorrect': pi_incorrect_var,
                    'ood': po_var}
    
    return prob_var_dict

def add_baseline(prob_var_dict,test_data,ood_test_data):
    ## Add baseline method    
    S = 10
    scale_n = 1
    scale_d = 0.1

    _,targets = next(iter(DataLoader(test_data,len(test_data))))
    _,ood_targets = next(iter(DataLoader(ood_test_data,len(ood_test_data))))

    base_method = torch.distributions.dirichlet.Dirichlet(torch.tensor([scale_d]*10))
    sample = base_method.sample(torch.Size([S,len(targets)]))
    sample_ood = base_method.sample(torch.Size([S,len(ood_targets)]))

    sample = (torch.randn((S,len(targets),10)) * scale_n).softmax(2)
    sample_ood = (torch.randn((S,len(targets),10)) * scale_n).softmax(2)

    pi_correct, pi_incorrect = sort_preds(sample,targets)

    max_index = pi_correct.mean(0).argmax(1)
    pi_correct_var = pi_correct[:,range(len(max_index)),max_index].var(0)

    max_index = pi_incorrect.mean(0).argmax(1)
    pi_incorrect_var = pi_incorrect[:,range(len(max_index)),max_index].var(0)

    max_index = sample_ood.mean(0).argmax(1)
    po_var = sample_ood[:,range(len(max_index)),max_index].var(0)

    prob_var_dict['BASE'] = {'id_correct': pi_correct_var,
                        'id_incorrect': pi_incorrect_var,
                        'ood': po_var}
    return prob_var_dict

def array_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return 'ERR'

def ax_violin(ax,var_dict,set_sigma=False, legend_true=False, fs = 12, title=None):
    ticks = list(var_dict.keys())
    
    id_correct_var_list = []
    id_incorrect_var_list = []
    ood_var_list = []
    for key in var_dict.keys():
        pi_correct = array_to_numpy(var_dict[key]['id_correct'])
        pi_incorrect = array_to_numpy(var_dict[key]['id_incorrect'])
        po = array_to_numpy(var_dict[key]['ood'])
        
        id_correct_var_list.append(pi_correct.tolist())
        id_incorrect_var_list.append(pi_incorrect.tolist())
        ood_var_list.append(po.tolist())

    id_vars_plot = ax.violinplot(id_correct_var_list,
                                positions=np.array(
        np.arange(len(id_correct_var_list)))*2-0.6, 
                                points=1000, widths=0.6,
                     showmeans=False, showextrema=False, showmedians=True)
    id_vars_plot_2 = ax.violinplot(id_incorrect_var_list,
                                positions=np.array(
        np.arange(len(id_incorrect_var_list)))*2, 
                                points=1000, widths=0.6,
                     showmeans=False, showextrema=False, showmedians=True)
    ood_vars_plot = ax.violinplot(ood_var_list,
                                positions=np.array(
        np.arange(len(ood_var_list)))*2+0.6,
                                points=1000, widths=0.6,
                     showmeans=False, showextrema=False, showmedians=True)
    
    for pc in id_vars_plot['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)

    for pc in id_vars_plot_2['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)

    for pc in ood_vars_plot['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)
    

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)

        if legend_true:    
            # use plot function to draw a small line to name the legend.
            plt.plot([], c=color_code, label=label)
            plt.legend(fontsize=fs)
    
    # # setting colors for each groups
    define_box_properties(id_vars_plot, '#008000', 'ID Correct')
    define_box_properties(id_vars_plot_2, '#D7191C', 'ID Incorrect')
    define_box_properties(ood_vars_plot, '#2C7BB6', 'OOD')
    
    ax.set_xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    ax.set_xlim(-1.2, len(ticks)*2 - 0.8)
    if set_sigma:
        ax.set_ylabel('$\sigma^2$', fontsize=fs)
    ax.tick_params(labelsize=fs)

    if title is not None:
        ax.set_title(title, fontsize=fs)

def plot_vmsp(prob_dict,title,save_fig):
    w, h, fs = 11, 6, 10
    fig, (ax1) = plt.subplots(1,1,facecolor='white', figsize = (1*w,1*h), sharey=True)
    fig.subplots_adjust(wspace=0)
    ax_violin(ax1,prob_dict, legend_true=True, fs=fs, title=title)
    ax1.set_yticks(np.array((0,0.15,0.3)))
    fig.savefig(save_fig,format='pdf',bbox_inches='tight')

def print_results(m, test_res, ei, train_res=None):
    ### print (train and) test prediction results
    print(f"\n--- Method {m} ---")
    if train_res is not None:
        print(f"Train Results -- Loss: {train_res[m]['nll'][ei]:.3f}; Train Acc: {train_res[m]['acc'][ei]:.1%}")
    t = time.strftime("%H:%M:%S", time.gmtime(test_res[m]['time'][ei]))
    s = f"Test Results -- Acc.: {test_res[m]['acc'][ei]:.1%}; ECE: {test_res[m]['ece'][ei]:.1%}; NLL: {test_res[m]['nll'][ei]:.3}; OOD-AUC: {test_res[m]['oodauc'][ei]:.1%}; AUC-ROC: {test_res[m]['aucroc'][ei]:.1%};"
    if m != 'MAP':
        s += f" VAR-ROC: {test_res[m]['varroc'][ei]:.1%}; VAR-ROC-ID: {test_res[m]['varroc_id'][ei]:.1%}; VAR-ROC-OOD: {test_res[m]['varroc_ood'][ei]:.1%}; VARROC-ROT: {test_res[m]['varroc_rot'][ei]:.1%};"
    s += f" Time h:m:s: {t}"
    print(s + "\n")



def predictions_tolerance(preds, targets, tau, deterministic=False):
    # Confidence tolerance
    if deterministic:
        conf = preds.max(-1)[0]
        conf_pred = preds[conf >= tau,:]
    else:
        conf = preds.mean(0).max(-1)[0]
        conf_pred = preds[:,conf >= tau,:]

    conf_targets = targets[conf >= tau]

    if not deterministic:
        var = preds.var(0).sum(-1); vm = var.max()
        # Var tolerance
        var_pred = preds[:,var < vm*(1-tau),:]
        var_targets = targets[var < vm*(1-tau)]

        return conf_pred, conf_targets, var_pred, var_targets
    else:
        return conf_pred, conf_targets 

def tolerance_acc(preds, test_loader, deterministic=False):
    targets = torch.cat([y for _,y in test_loader])

    tau_range = torch.linspace(0,1,100)
    acc_conf = []
    if not deterministic:
        acc_var = []

    for tau in tau_range:
        pred_target = predictions_tolerance(preds, targets, tau, deterministic)
        # conf_pred, conf_targets, var_pred, var_targets = predictions_tolerance(preds, targets, tau)
        if deterministic:
            conf_pred, conf_targets = pred_target
            acc_conf.append((conf_pred.argmax(-1) == conf_targets).type(torch.float).mean().item())
        else:
            conf_pred, conf_targets, var_pred, var_targets = pred_target 
            acc_conf.append((conf_pred.mean(0).argmax(-1) == conf_targets).type(torch.float).mean().item())
            acc_var.append((var_pred.mean(0).argmax(-1) == var_targets).type(torch.float).mean().item())

    if deterministic:
        return acc_conf
    else:
        return acc_conf, acc_var
    
def tolerance_plot(acc_dict, save_fig):
    f,(ax) = plt.subplots(1,2, figsize = (10,4))
    f.subplots_adjust(wspace=0.3)

    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel(r"$\tau$")

    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel(r"$\tau$")

    for m in acc_dict.keys():
        for idx,k in enumerate(acc_dict[m].keys()):
            acc = acc_dict[m][k]
            ax[idx].plot(np.linspace(0,1,len(acc)), acc, label=m)

    ax[0].legend()
    ax[1].legend()

    f.savefig(save_fig,format='pdf',bbox_inches='tight')

def rotated_dataset(id_predictions_probit, pred_func, dataset):
    '''
    Input:
        id_predictions_probit: predictions on MNIST test set, size S x N x C, in PROBIT space. 

        pred_func: (function) that takes dataset (torch.utils.data.Dataset), and outputs S x N x C in logits space, where
                    S is number of mc samples, N is len(dataset), and C is number of classes

        dataset: (str) in ['mnist','fmnist']

    Output:
        Averaged VARROC over all rotations of MNIST. Higher is better.
    
    '''
    if dataset == 'mnist':
        dataset_func = ds.get_rotated_MNIST
    elif dataset == 'fmnist':
        dataset_func = ds.get_rotated_FMNIST
    else:
        print('Invalid dataset choice. Valid choices are [mnist, fmnist].')

    angles = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    varroc = 0
    acc = []
    for angle in angles:
        dataset_rot = dataset_func(angle)
        rot_loader = DataLoader(dataset_rot, 100)
        rot_targets = torch.cat([y for _,y in rot_loader])
        predictions_rot = pred_func(dataset_rot) # S x N x C, logits
        predictions_rot = predictions_rot.softmax(-1)
        score = ood_auc(id_predictions_probit.var(0).sum(-1).detach().cpu(), predictions_rot.var(0).sum(-1).detach().cpu())
        varroc += score
        acc.append((predictions_rot.mean(0).argmax(-1) == rot_targets).type(torch.float).mean().item())
    varroc /= len(angles)
    return varroc, acc