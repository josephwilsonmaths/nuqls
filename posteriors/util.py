import torch
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def evaluate_batch(dataset,model,batch_size):
            if batch_size > 0:
                bs = batch_size
            elif batch_size == 0:
                bs = len(dataset)
            loader_batched = DataLoader(dataset,bs)
            evaluate = []
            for b,(x,_) in enumerate(loader_batched):
                evaluate.append(model(x.to(device)).detach())
            evaluate = torch.cat(evaluate,dim=0)
            return evaluate

def evaluate_batch_T(dataset,model,batch_size):
            if batch_size > 0:
                bs = batch_size
            elif batch_size == 0:
                bs = len(dataset)
            loader_batched = DataLoader(dataset,bs)
            evaluate = []
            for b,(x,_) in enumerate(loader_batched):
                evaluate.append(model(x.to(device)).detach()) # output is (samples x batchsize x output)
            evaluate = torch.cat(evaluate,dim=1)
            return evaluate

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
            pred_i = evaluate_batch(dataset=dataset,model=models[i],batch_size=bs)
            ensemble_pred[i,:,:] = pred_i
        ensemble_pred_avg = ensemble_pred.mean(axis=0)
        ensemble_pred_var = ensemble_pred.var(axis=0)
        return ensemble_pred_avg, ensemble_pred_var

def ensemble_sampler(dataset,M,models,n_output,bs):
        ensemble_pred = torch.empty((M,len(dataset),n_output),device='cpu')
        for i in range(M):
            pred_i = evaluate_batch(dataset=dataset,model=models[i],batch_size=bs)
            ensemble_pred[i,:,:] = pred_i.detach()
        de_f = ensemble_pred.softmax(dim=2).mean(0)
        de_var = ensemble_pred.softmax(dim=2).var(0)
        return de_f, de_var, ensemble_pred.softmax(dim=2)

def swag_sampler(dataset,model,T,n_output,bs):
    swag_pred_samples = torch.empty((T,len(dataset),n_output),device='cpu')
    for t in range(T):
        model.sample(cov=True)
        pred_i = evaluate_batch(dataset=dataset, model=model, batch_size=bs)
        swag_pred_samples[t] = pred_i
    swag_probs = swag_pred_samples.softmax(dim=2).mean(axis=0)
    swag_var = swag_pred_samples.softmax(dim=2).var(axis=0)
    return swag_probs, swag_var, swag_pred_samples.softmax(dim=2)


def dropout_active(m):
    if isinstance(m,torch.nn.Dropout):
        m.train()

def dropout_sampler(dataset, model, T, n_output, bs, logits=False):
    model.eval()
    model.apply(dropout_active)
    dropout_pred_samples = torch.empty((T,len(dataset),n_output),device='cpu')
    for t in range(T):
        mc_pred = evaluate_batch(dataset=dataset,model=model,batch_size=bs)
        dropout_pred_samples[t,:,:] = mc_pred
    if logits:
        return dropout_pred_samples.mean(axis=0), dropout_pred_samples.var(axis=0), dropout_pred_samples
    else:
        return dropout_pred_samples.softmax(dim=2).mean(axis=0), dropout_pred_samples.softmax(dim=2).var(axis=0), dropout_pred_samples.softmax(dim=2)
    
def lla_sampler(dataset, model, bs):
    lla_pred_samples = evaluate_batch_T(dataset=dataset,model=model,batch_size=bs)
    lla_pred_samples = lla_pred_samples.cpu()
    return lla_pred_samples.mean(axis=0), lla_pred_samples.var(axis=0), lla_pred_samples