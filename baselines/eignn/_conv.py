from scipy.linalg import eig, eigh
import psutil
import os

import torch
import torch.sparse
from torch.nn import Module, Parameter

from _deq import IDMFunction


class IDM_SGC(Module):
    def __init__(self, adj, sp_adj, m, num_eigenvec, gamma, adj_preload_file=None):
        super(IDM_SGC, self).__init__()
        self.F = Parameter(torch.FloatTensor(m, m), requires_grad=True)
        self.S = adj
        self.gamma = Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        sy = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        process = psutil.Process(os.getpid())
        mem_info_before = process.memory_info()
        rss_before = mem_info_before.rss
        if sy:
            self.Lambda_S, self.Q_S = eigh(sp_adj.toarray())
        else:
            self.Lambda_S, self.Q_S = eig(sp_adj.toarray())
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)
        # Get the memory usage after the operations
        mem_info_after = process.memory_info()
        rss_after = mem_info_after.rss
        mem_increase = rss_after - rss_before
        print(f"Memory increased by {mem_increase / (1024 * 1024)} MB")
        exit()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        return IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)