import torch.nn as nn
import torch.nn.functional as F

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(3*32*32, 10)
    def forward(self, xb): 
        xb = xb.reshape(-1, 3*32*32)
        out = self.linear(xb)
        return out

class TwoLayer(nn.Module):
    def __init__(self):
        super(TwoLayer, self).__init__()
        self.fc1 = nn.Linear(3*32*32,120)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(120,10)
    def forward(self, xb):
        xb = xb.reshape(-1, 3*32*32)
        out = self.fc1(xb)
        out = self.act(out)
        out = self.fc2(out)
        return out
    
class TwoLayerBottle(nn.Module):
    def __init__(self):
        super(TwoLayerBottle, self).__init__()
        self.fc1 = nn.Linear(3*32*32,2)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(2,10)
    def forward(self, xb):
        xb = xb.reshape(-1, 3*32*32)
        out = self.fc1(xb)
        out = self.act(out)
        out = self.fc2(out)
        return out