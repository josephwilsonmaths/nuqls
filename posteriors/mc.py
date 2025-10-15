import torch
import tqdm
import utils.training
import utils.optimizers
import copy


class MCDropout(object):
    def __init__(
        self,
        network: torch.nn.Module,
        p: float = 0.1
        ):
        self.original_network = network
        self.device = next(self.original_network.parameters()).device
        self.p = p

        if network._get_name() == 'ResNet':
            self.network = self.original_network
            self.network.p = self.p
            self.network.dropout = torch.nn.Dropout(self.p)
        else:
            child_list = list(self.original_network.children())
            if len(child_list) > 1:
                child_list = child_list
            elif len(child_list) == 1:
                child_list = child_list[0]
            first_layers = child_list[:-1]
            final_layer = child_list[-1]
            first_layers.append(torch.nn.Dropout(self.p))
            first_layers.append(final_layer)
            self.network = torch.nn.Sequential(*first_layers)
    
    def test(self, loader, samples, verbose=False):
        self.network.train()

        prediction_samples = []

        if verbose:
            pbar = tqdm.trange(samples)
        else:
            pbar = range(samples)
        for s in pbar:
            batches = []
            for x,_ in loader:
                batches.append(self.network(x.to(self.device)).detach()) # output is (batchsize x output)
            batches = torch.cat(batches, dim=0)
            prediction_samples.append(batches)
        predictions = torch.stack(prediction_samples, dim=0)

        return predictions # M x N x C
    
    def mean_variance(self, loader, samples, verbose=False):
        predictions = self.test(loader, samples, verbose).cpu()
        return predictions.softmax(-1).mean(0), predictions.softmax(-1).var(0)