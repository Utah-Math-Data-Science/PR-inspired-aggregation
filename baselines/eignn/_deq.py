import torch
from torch.autograd import Function

epsilon_F = 10**(-12)
def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1/(FF_norm+epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G

class IDMFunction(Function):
    @staticmethod
    def forward(ctx, X, F, S, Q_S, Lambda_S, gamma):
        Lambda_F, Q_F = torch.symeig(g(F), eigenvectors=True)
        Lambda_F = Lambda_F.view(-1,1)
        G = get_G(Lambda_F, Lambda_S, gamma)
        Z = Q_F @ (G * (Q_F.t() @ X @ Q_S)) @ Q_S.t()
        ctx.save_for_backward(F, S, Q_F, Q_S, Z, G, X, gamma)
        return Z

    @staticmethod
    def backward(ctx, grad_output):
        grad_Z = grad_output
        F, S, Q_F, Q_S, Z, G, X, gamma = ctx.saved_tensors
        FF = F.t() @ F
        FF_norm = torch.norm(FF, p='fro')
        R = G * (Q_F.t() @ grad_Z @ Q_S)
        R = Q_F @ R @ Q_S.t() @ torch.sparse.mm(S, Z.t())
        scalar_1 = gamma * (1/(FF_norm+epsilon_F))
        scalar_2 = torch.sum(FF * R)
        scalar_2 = 2 * scalar_2 * (1/(FF_norm**2 + epsilon_F * FF_norm))
        grad_F = (R + R.t()) - scalar_2 * FF
        grad_F = scalar_1 * (F @ grad_F)
        grad_X = None
        return grad_X, grad_F, None, None, None, None