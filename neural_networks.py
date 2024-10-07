import torch

class oneLayerMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output_size = 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1)
        )

    def forward(self, x):
        output = self.net(x)
        return output
    
class twoLayerMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output_size = 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width,1)
        )

    def forward(self, x):
        output = self.net(x)
        return output

class oneLayerMLP_Heteroskedastic(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,width)
        self.relu = torch.nn.ReLU()
        self.linear_mu = torch.nn.Linear(width,1)
        self.linear_sig = torch.nn.Linear(width,1)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        mu = self.linear_mu(x)
        variance = self.linear_sig(x)
        variance = torch.nn.functional.softplus(variance) + 1e-6
        return mu, variance
    
class twoLayerMLP_Heteroskedastic(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,width)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(width,width)
        self.linear_mu = torch.nn.Linear(width,1)
        self.linear_sig = torch.nn.Linear(width,1)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        mu = self.linear_mu(x)
        variance = self.linear_sig(x)
        variance = torch.nn.functional.softplus(variance) + 1e-6
        return mu, variance
    
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight,mean=0,std=1)
        torch.nn.init.normal_(m.bias,mean=0,std=1)