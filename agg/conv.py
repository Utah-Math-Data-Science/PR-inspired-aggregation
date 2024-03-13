from typing import Optional
from torch import Tensor

import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj

from functions import MonotoneNonlinear
from solvers import FixedPointSolver

class ImplicitLayer(MessagePassing):
    def __init__(self,
        nonlin_module: MonotoneNonlinear,
        bias_module: nn.Module,
        phantom_grad: int = 0,
        beta_init: float = 0.,
        gamma_init: float = 0.,
        tol: float = 1e-6,
        max_iter: int = 50,
        sigmoid: bool = True,
        fix_params: bool = False,
        valid_u0: bool = True,
    ) -> None:
        super(ImplicitLayer, self).__init__()
        self.bias = bias_module
        self.nonlin = nonlin_module
        self.phantom_grad = phantom_grad

        # self.log_beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float))
        self.valid_u0 = valid_u0
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float), requires_grad=not fix_params)
        self.tol = tol
        self.act = nn.Sigmoid() if sigmoid else nn.ReLU()

        self.pos_gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float), requires_grad=not fix_params)
        self.max_iter = max_iter
        self.u0 = None

        pass

    def forward(self, x, edge_index, edge_weight, u0=None, bias_args=None):

        # You may store u0 in this object (ideal for static graphs)
        # or pass it as an argument (ideal for dynamic graph)
        u0 = u0 if u0 is not None else self.u0 

        self.gamma = (1 + torch.abs(2*torch.sigmoid(self.beta)-1)) + self.act(self.pos_gamma) #torch.exp(self.log_gamma) + 2 * torch.nn.functional.relu(torch.exp(self.log_beta) -0.5) + 1

        if self.phantom_grad: #moved inside of FP
            with torch.no_grad():
                u, itr = self.iterate(x, edge_index, edge_weight, max(0,self.max_iter-self.phantom_grad), u0=u0, bias_args=bias_args)
            for _ in range(self.phantom_grad):
                u.requires_grad_(True)
                u, _ = self.iterate(x, edge_index, edge_weight, 1, u0=u, bias_args=bias_args)
        else:
                u, itr = self.iterate(x, edge_index, edge_weight, self.max_iter, u0=u0, bias_args=bias_args)
        self.itr = itr
        if (self.training) or (self.valid_u0): self.u0 = u.clone().detach()
        z = self.nonlin(u)
        return z


    def V(self, z, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=z, edge_weight=edge_weight)
        # new_agg = ((0.5+torch.exp(self.log_beta))*z - (torch.exp(self.log_beta) -0.5) * agg) / self.gamma
        new_agg = (z + (2*torch.sigmoid(self.beta) - 1) * agg) / self.gamma 
        return new_agg

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1,1) * x_j
    
    def iterate(self, x, edge_index, edge_weight, max_iters, u0: Optional[Tensor] = None, bias_args=None):
        u = u0 if u0 is not None else torch.rand_like(x, requires_grad=True)
        err, itr = 1e30, 0
        bias = self.bias(x) if bias_args is None else self.bias(x, **bias_args)
        while (err > self.tol and itr < max_iters and not np.isnan(err)):
            u_half = (2*self.nonlin(u) - u - bias)
            un = 2*self.V(u_half, edge_index, edge_weight)-2*self.nonlin(u)+u
            err = torch.norm(u-un, np.inf).item()
            itr = itr + 1
            u = un

        return u, itr


    # TODO: If we don't use phantom we should use this.
    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, z):
            ctx.splitter = splitter
            ctx.save_for_backward(z)
            return z
        @staticmethod
        def backward(ctx, g):
            return None, None