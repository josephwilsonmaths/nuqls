import torch
import copy
import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# BSD 2-Clause License

# Copyright (c) 2019, Wesley Maddox, Timur Garipov, Pavel Izmailov,  Dmitry Vetrov, Andrew Gordon Wilson
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def swag_parameters(module, params, cov_mat=True):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        if cov_mat is True:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        params.append((module, name))

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


class SWAG(torch.nn.Module):

    def __init__(self,net,epochs=50,lr=1e-3,cov_mat=True,max_num_models=0,var_clamp=1e-30,wd=0,target='multiclass'):
        super(SWAG,self).__init__()
        self.swag_net = copy.deepcopy(net)

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = []

        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.cov_mat = cov_mat
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        self.target = target
        if self.target == 'multiclass':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif self.target == 'binary':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.swag_net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.swag_net.apply(
                    lambda module: swag_parameters(
                        module=module, params=self.params, cov_mat=self.cov_mat
                    )
                )
        
    def print_layers(self):
        for i,(module,name) in enumerate(self.params):
            print((module,name))
        # for (np,p),i in zip(self.swag_net.named_parameters(),range(10)):
            # print(np,i)

    def forward(self, x):
        # self.swag_net.to(device)
        return self.swag_net(x)
        
    def train_swag(self,train_dataloader, progress_bar=True):
        if progress_bar:
            pbar = tqdm.trange(self.epochs)
        else:
            pbar = range(self.epochs)
        for _ in pbar:
            if self.target == 'multiclass':
                train_loop(dataloader=train_dataloader, model=self.swag_net, loss_fn=self.loss_fn,
                            optimizer=self.optimizer, scheduler=None)
            elif self.target == 'binary':
                train_loop_binary(dataloader=train_dataloader, model=self.swag_net, loss_fn=self.loss_fn,
                            optimizer=self.optimizer, scheduler=None)
            self.collect_model()
        
    def collect_model(self):
        for (module,name), param in zip(self.params,self.swag_net.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + param.detach() / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + param.detach() ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.cov_mat is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (param.detach() - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)       

    def sample(self, scale=1.0, cov=False, seed=None, blockwise=False):
        if seed is not None:
            torch.manual_seed(seed)
        if not blockwise:
            self.sample_fullrank(scale, cov)
        else:
            self.sample_blockwise(scale, cov)
        self.swag_net.to(device)

    def sample_blockwise(self, scale, cov):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                w = mean + scaled_diag_sample + cov_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, torch.nn.parameter.Parameter(w))

    def sample_fullrank(self, scale, cov):
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, torch.nn.parameter.Parameter(sample))

    def test(self, loader, samples):

        prediction_samples = []
        for _ in tqdm.trange(samples):
            self.sample(cov=True)
            batches = []
            for x,_ in loader:
                batches.append(self.forward(x=x).detach()) # output is (batchsize x output)
            batches = torch.cat(batches, dim=0)
            prediction_samples.append(batches)
        prediction_samples = torch.stack(prediction_samples, dim=0)

        return prediction_samples # Samples x N x C

class SWAG_R(torch.nn.Module):
    '''
    SWAG implementation for regression.
    '''
    def __init__(self,net,epochs=50,lr=1e-3,cov_mat=True,max_num_models=0,var_clamp=1e-30):
        super(SWAG_R,self).__init__()
        self.swag_net = copy.deepcopy(net)

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = []

        self.epochs = epochs
        self.lr = lr
        self.cov_mat = cov_mat
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        self.loss_fn = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.swag_net.parameters(), lr=self.lr)

        self.swag_net.apply(
                    lambda module: swag_parameters(
                        module=module, params=self.params, cov_mat=self.cov_mat
                    )
                )
        
    def print_layers(self):
        print("\n=== Printing parameters of self.swag_net ===")
        for keys in self.swag_net.state_dict().keys():
                print(f'{keys} : {self.swag_net.state_dict()[keys].device}')

    def forward(self, x):
        return self.swag_net(x)
    
    def train_loop(self,train_loader,optimizer):
        # Compute prediction error
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = self.swag_net(x)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
    def train_swag(self,train_loader, weight_decay):
        # optimizer = torch.optim.SGD(self.swag_net.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(self.swag_net.parameters(), lr=self.lr, weight_decay=weight_decay)
        for _ in range(self.epochs):
            self.train_loop(train_loader,optimizer)
            self.collect_model()
        
    def collect_model(self):
        for (module,name), param in zip(self.params,self.swag_net.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + param.detach() / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + param.detach() ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.cov_mat is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (param.detach() - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)       

    def sample(self, scale=1.0, cov=False, seed=None, blockwise=False):
        if seed is not None:
            torch.manual_seed(seed)
        if not blockwise:
            self.sample_fullrank(scale, cov)
        else:
            self.sample_blockwise(scale, cov)

    def sample_blockwise(self, scale, cov):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                w = mean + scaled_diag_sample + cov_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, torch.nn.parameter.Parameter(w))

        self.swag_net.to(device)

    def sample_fullrank(self, scale, cov):
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, torch.nn.parameter.Parameter(sample))

        self.swag_net.to(device)

def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
        model.train()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)    
        train_loss, correct = 0, 0
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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss /= num_batches
        correct /= size

        return train_loss, correct

def train_loop_binary(dataloader, model, loss_fn, optimizer, scheduler, device='cpu', train_mode=True):
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
        loss = loss_fn(pred, y.to(dtype=torch.float64).unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        correct += (pred.sigmoid().round().squeeze(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    correct /= size

    return train_loss, correct