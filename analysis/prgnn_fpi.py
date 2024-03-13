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
from torch.nn import Linear, Module, Sequential, Tanh
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything

sys.path.append('/root/workspace/PR-inspired-aggregation/agg/')
from conv import ImplicitLayer
from functions import MonotoneNonlinear, ReLU, TanH
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
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: int,
        phantom_grad: int,
        beta_init: float,
        gamma_init: float,
        tol: float,
        max_iter: int = 50,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.act = ReLU()
        self.enc = Linear(in_channels, hidden_channels, bias=False)
        self.dec = Linear(hidden_channels, out_channels, bias=False)

        nonlin_module = ReLU()
        bias_module = Sequential(Linear(hidden_channels, hidden_channels, bias=False), Tanh(), Linear(hidden_channels, hidden_channels, bias=False))
        self.igl = ImplicitLayerwStorage(nonlin_module, bias_module, phantom_grad=phantom_grad, beta_init=beta_init, gamma_init=gamma_init, tol=tol, max_iter=max_iter, sigmoid=False)
        pass

    def forward(self, x, edge_index, edge_weight):
        x = self.enc(x)
        x = F.dropout(x, self.dropout, training=self.training)
        self.igl.storage = []
        x = self.act( self.igl(x, edge_index, edge_weight) )
        x = self.dec(x)
        return x

class ImplicitLayerwStorage(ImplicitLayer):
    def __init__(self,
        nonlin_module: MonotoneNonlinear,
        bias_module: Module,
        phantom_grad: int = 0,
        beta_init: float = 0.,
        gamma_init: float = 0.,
        tol: float = 1e-6,
        max_iter: int = 50,
        sigmoid: bool = True
    ) -> None:
        super(ImplicitLayerwStorage, self).__init__(nonlin_module, bias_module, phantom_grad, beta_init, gamma_init, tol, max_iter, sigmoid)
        self.storage = []

    def iterate(self, x, edge_index, edge_weight, max_iters, u0: Optional[Tensor] = None):
        u = u0 if u0 is not None else torch.zeros_like(x, requires_grad=True)
        err, itr = 1e30, 0
        self.storage.append(u)
        while (err > self.tol and itr < max_iters and not np.isnan(err)):
            u_half = (2*self.nonlin(u) - u - self.bias(x))
            un = 2*self.V(u_half, edge_index, edge_weight)-2*self.nonlin(u)+u
            err = torch.norm(u-un, np.inf).item()
            itr = itr + 1
            u = un
            self.storage.append(u)
        return u, itr



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
        chain_len=20,
        num_chains=20,
        num_classes=2,
        feature_dim=100,
        noise=0.0,
    )
    # Load model
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = data.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = data.num_classes,
        dropout = model_kwargs['dropout'],
        phantom_grad = model_kwargs['phantom_grad'],
        beta_init = model_kwargs['beta_init'],
        gamma_init= model_kwargs['gamma_init'],
        tol = model_kwargs['tol'],
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
            'fwd_itr: {:03d}, '
            'beta: {:.3f}, '
            'gamma: {:.3f}, '
            'train_loss: {:.7f}, '
            'train_acc: {:2.2f}, '
            ''.format(epoch+1, model.igl.itr, 2*torch.sigmoid(model.igl.beta).item()-1, model.igl.gamma.item(), train_loss, 100*train_acc))

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

    run_training(cfg, data, model, epochs=10)
    first_fpi = model.igl.storage
    # run_training(cfg, data, model, epochs=1)
    _, _ = validate(cfg, data, model)
    second_fpi = model.igl.storage
    run_training(cfg, data, model, epochs=1)
    third_fpi = model.igl.storage
    _, _ = validate(cfg, data, model)
    fourth_fpi = model.igl.storage

    fpi_1 = torch.stack([val[0,:2] for val in first_fpi]).detach().cpu().numpy()
    fpi_2 = torch.stack([val[0,:2] for val in second_fpi]).detach().cpu().numpy()
    fpi_3 = torch.stack([val[0,:2] for val in third_fpi]).detach().cpu().numpy()
    fpi_4 = torch.stack([val[0,:2] for val in fourth_fpi]).detach().cpu().numpy()

    plt.scatter(fpi_1[:,0], fpi_1[:,1], alpha=np.linspace(0.1,1,len(fpi_1)), color='b')
    print(first_fpi[0].shape, fpi_1.shape)
    plt.scatter(fpi_2[:,0], fpi_2[:,1], alpha=np.linspace(0.1,1,len(fpi_1)), color='r')
    plt.scatter(fpi_3[:,0], fpi_3[:,1], alpha=np.linspace(0.1,1,len(fpi_1)), color='b')
    plt.scatter(fpi_4[:,0], fpi_4[:,1], alpha=np.linspace(0.1,1,len(fpi_1)), color='r')
    plt.savefig(f'/root/workspace/out/prgnn_fpi.png')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_chains()