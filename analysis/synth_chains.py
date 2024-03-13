#!/usr/bin/python3
from typing import Optional
from torch import Tensor

import os
import random
import os

import hydra
from omegaconf import OmegaConf
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
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
        beta: float,
        tol: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.act = ReLU()
        self.enc = Linear(in_channels, hidden_channels, bias=False)
        self.dec = Linear(hidden_channels, out_channels, bias=False)

        nonlin_module = ReLU()
        bias_module = Linear(hidden_channels, hidden_channels, bias=False)
        self.igl = ImplicitLayerwStorage(nonlin_module, bias_module, phantom_grad=phantom_grad, beta=beta, tol=tol)
        pass

    def forward(self, x, edge_index, edge_weight):
        x = self.enc(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act( self.igl(x, edge_index, edge_weight) )
        x = self.dec(x)
        return x

class ImplicitLayerwStorage(ImplicitLayer):
    def __init__(self,
        nonlin_module: MonotoneNonlinear,
        bias_module: Module,
        phantom_grad: int = 0,
        beta: float = -.1,
        tol: float = 1e-6
    ) -> None:
        super(ImplicitLayerwStorage, self).__init__(nonlin_module, bias_module, phantom_grad, beta, tol)

        self.storage = []

    def iterate(self, x, edge_index, edge_weight, max_iters, u0: Optional[Tensor] = None):
        u = u0 if u0 is not None else torch.zeros_like(x, requires_grad=True)
        err, itr = 1e30, 0
        while (err > self.tol and itr < max_iters and not np.isnan(err)):
            self.storage.append(u)
            u_half = (2*self.nonlin(u) - u - self.alpha*self.bias(x))
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
        split=args['split'],
        chain_len=cfg.data['chain_len'],
        num_chains=cfg.data['num_chains'],
        num_classes=cfg.data['num_classes'],
        feature_dim=cfg.data['feature_dim'],
        noise=cfg.data['noise'],
    )
    # Load model
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = data.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = data.num_classes,
        dropout = model_kwargs['dropout'],
        phantom_grad = model_kwargs['phantom_grad'],
        beta = model_kwargs['beta'],
        tol = model_kwargs['tol'],
    )
    # Load Model
    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, data

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------


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


#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/analysis/", config_name="synth_chains.yaml")
def run_chains(cfg):
    # Initialize settings to wandb server
    wandb.init(entity='utah-math-data-science',
                project='pr-inspired-aggregation',
                mode='disabled',
                name='ignn-chains',
                dir='/root/workspace/out/',
                tags=['synth-chains', 'ignn'],
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, data= load(cfg)
    print(model)
    model.to(cfg.setup['device'])
    data.to(cfg.setup['device'])

    test_loss, test_acc = test(cfg, data, model)
    # print(len(model.igl.storage), model.igl.storage[0].shape)
    # make figure
    chain_len = cfg.data['chain_len']
    num_chains = cfg.data['num_chains']
    num_classes = cfg.data['num_classes']
    # for k in [0,11]:
    #     u = model.igl.storage[k]
    #     plt.figure()
    #     node0, node1 = [], []
    #     for n in range(model.igl.storage[0].shape[0]):
    #         node = u[n].detach().cpu().numpy()
    #         # print(node)
    #         # for node in u[n*chain_len:(n+1)*chain_len]:
    #         node0.append(node[0])
    #         node1.append(node[1])
    #     plt.plot(node0,node1, 'o-')
    #     plt.savefig(f'/root/workspace/out/ignn-chain-{k}.png')
    #     plt.close()
    print(model.igl.u0.shape)
    plt.imshow(model.igl.u0.view(chain_len*num_classes,-1).detach().cpu().numpy())
    plt.savefig(f'/root/workspace/out/ignn-chain.png')
    print(f'Test Accuracy: {100*test_acc}')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_chains()