import math
import random
import re
from torch_sparse import spmm
import torch
import torch.nn as nn


"""
Monotone Nonlinear Functions
"""
class MonotoneNonlinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

class ReLU(MonotoneNonlinear):
    def forward(self, z):
        return torch.relu(z)
    def derivative(self, z):
        # return (z > torch.finfo(z.dtype).eps).type_as(z)
        return (z > 0).type_as(z)

class Ident(MonotoneNonlinear):
    def forward(self, z):
        return z
    def derivative(self, z):
        return torch.ones(z.shape,dtype=z.dtype,device=z.device)

class TanH(MonotoneNonlinear):
    def __init__(self,**kwargs):
        super().__init__()
    def forward(self, z):
        return torch.tanh(z)
    def derivative(self, z):
        return torch.ones_like(z)-torch.tanh(z)**2
    def inverse(self, z):
        return torch.arctanh(z)