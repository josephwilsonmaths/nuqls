import numpy as np
import sklearn.metrics as sk

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