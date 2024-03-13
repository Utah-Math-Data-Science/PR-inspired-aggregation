#!/usr/bin/python3
from typing import Optional
from torch import Tensor

import os
import random
import sys

import hydra
from omegaconf import OmegaConf
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear, Module, Sequential, Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_dense_adj

sys.path.append('/root/workspace/PR-inspired-aggregation/baselines/ignn/')
from _deq import ImplicitFunction
from _conv import ImplicitLayer
sys.path.append('/root/workspace/PR-inspired-aggregation/tasks/datasets/')
from synthetic import synth_chains_data

seed_everything(0)

"""
Synthetic Chains Dataset
    Split: 5/10/85
"""

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------

class Model(Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            out_channels,
            num_nodes,
            dropout,
            kappa,
        )-> None:
        super(Model, self).__init__()

        #one layer with V
        self.igl = ImplicitLayerwStorage(in_channels, hidden_channels, num_nodes, kappa=kappa)
        self.dropout = dropout
        self.X_0 = torch.zeros(hidden_channels, num_nodes)
        self.V = Linear(hidden_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_weight):
        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0]

        x = self.igl(self.X_0.to(x.device), adj, x.T, F.relu, A_orig=adj).T
        self.X_0 = x.detach().T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V(x)
        return x


class ImplicitLayerwStorage(ImplicitLayer):
    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitLayerwStorage, self).__init__(in_features, out_features, num_node, kappa, b_direct)
        self.storage = []
        self.X0 = None

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = self.projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = torch.mm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.mm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.mm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 #+ support_2
        X_new, storage = ImplicitFunctionwStorage.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)
        self.storage = storage
        return X_new
    
class ImplicitFunctionwStorage(ImplicitFunction):
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D, storage = ImplicitFunctionwStorage.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X, storage

    @staticmethod
    def inn_pred(W, X, A, B, phi, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False):
        # TODO: randomized speed up
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        if not trasposed_A: storage = [X]
        #X = B if X is None else X

        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            X_new = phi(support + B)
            if not trasposed_A: storage.append(X_new)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi, storage


#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------------------------------------------------------

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

#----------------------------------------------------------------------------------------------------------------------------------------------------

def clip_gradient(model, clip_norm=10):
    """ clip gradients of each parameter by norm """
    for param in model.parameters():
        torch.nn.utils.clip_grad_norm(param, clip_norm)
    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Config/Model/Dataset
#----------------------------------------------------------------------------------------------------------------------------------------------------

def setup(cfg):
    # Set device
    args = cfg.setup
    cfg['setup']['device'] = args['device'] if torch.cuda.is_available() else 'cpu'
    os.environ["WANDB_DIR"] = os.path.abspath(args['wandb_dir'])
    # Change file name for sweeping *Prior to setting seed*
    if args['sweep']:
        run_id = wandb.run.id
        cfg['load']['checkpoint_path']=cfg['load']['checkpoint_path'][:-3]+str(run_id)+'.pt'
    # Set Backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Set Seed
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    set_seed(args['seed'])
    pass

#----------------------------------------------------------------------------------------------------------------------------------------------------

def load(cfg):
    args = cfg.load
    # Load data
    data = synth_chains_data(
        split='fixed_05/10/85',
        chain_len=100,
        num_chains=20,
        num_classes=10,
        feature_dim=100,
        noise=0.0,
    )
    # Load model
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = data.num_features,
        hidden_channels = 16,
        out_channels = data.num_classes, 
        num_nodes = data.num_nodes,
        dropout = 0.0,
        kappa = 0.9,
    )
    return model, data

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    return loss.item(), acc

def validate(cfg, data, model):
    model.eval()
    output = model(data.x, data.edge_index, data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    
    acc = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss.item(), acc


def test(cfg, data, model):
    model.eval()
    output = model(data.x, data.edge_index, data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    
    acc = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss.item(), acc

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model, epochs=100):
    args = cfg.train
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    for epoch in range(epochs):
        train_loss, train_acc = train(cfg, data, model, optimizer)

        print('Epoch: {:03d}, '
            'train_loss: {:.7f}, '
            'train_acc: {:2.2f}, '
            ''.format(epoch+1, train_loss, 100*train_acc))

    return 1




#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/analysis/", config_name="synth_chains.yaml")
def run_chains(cfg):
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, data= load(cfg)
    print(model)
    model.to(cfg.setup['device'])
    data.to(cfg.setup['device'])

    for i in range(20):
        run_training(cfg, data, model, epochs=1)
        fpi = model.igl.storage
        fpi = torch.stack([val[0,:2] for val in fpi]).detach().cpu().numpy()
        plt.scatter(fpi[:,0], fpi[:,1], alpha=np.linspace(0.1,1,len(fpi)), color='b')
        ax = plt.gca()
        ax.annotate(str(i), xy=fpi[-1,:], textcoords='offset points', xytext=(0,5), ha='center', fontsize=8)
    plt.savefig(f'/root/workspace/out/ignn_fpi.png')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_chains()