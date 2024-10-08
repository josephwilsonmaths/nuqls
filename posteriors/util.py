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