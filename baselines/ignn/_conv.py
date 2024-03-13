import math
import numpy as np

import torch
import torch.sparse
from torch.nn import Module, Parameter

from _deq import ImplicitFunction

class ImplicitLayer(Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitLayer, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = self.projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = torch.mm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.mm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.mm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 #+ support_2
        return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)

    def projection_norm_inf(self, A, kappa=0.99, transpose=False):
        """ project onto ||A||_inf <= kappa return updated A"""
        # TODO: speed up if needed
        v = kappa
        if transpose:
            A_np = A.T.clone().detach().cpu().numpy()
        else:
            A_np = A.clone().detach().cpu().numpy()
        x = np.abs(A_np).sum(axis=-1)
        for idx in np.where(x > v)[0]:
            # read the vector
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                # proposal: alpha <= a[i]
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # verify
            assert np.isclose(np.abs(a).sum(), v, atol=1e-4)
            # write back
            A_np[idx, :] = a
        A.data.copy_(torch.tensor(A_np.T if transpose else A_np, dtype=A.dtype, device=A.device))
        return A

