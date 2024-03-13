#!/usr/bin/python3
from typing import Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor

import json
import os
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
import random
import torch
from torch.nn import BatchNorm1d, Linear, Module
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.transforms as T

from ogb.nodeproppred import Evaluator
from datasets.arxiv_dataset import arxiv_dataloaders
sys.path.append('/root/workspace/PR-inspired-aggregation/agg/')
from conv import ImplicitLayer
from functions import ReLU

"""
arXiv Dataset
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
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.L_in = Linear(in_channels, hidden_channels, bias=False)
        self.act = ReLU()
        self.enc = Linear(in_channels, hidden_channels)
        self.dec = Linear(hidden_channels, out_channels)

        nonlin_module = ReLU()
        bias_module = Linear(hidden_channels, hidden_channels, bias=False)
        self.igl = ImplicitLayer(nonlin_module, bias_module, phantom_grad=phantom_grad, beta_init=beta_init, gamma_init=gamma_init, tol=tol)
        self.batch_norm = BatchNorm1d(hidden_channels)
        pass

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.enc(x)
        #x = self.act( self.batch_norm( self.igl(x, edge_index, edge_weight) ) )
        x = self.act( self.igl(x, edge_index, edge_weight) )
        x = self.dec(x)
        return x


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
    # Set Transforms
    transform = T.NormalizeFeatures()
    # Load Dataset
    dataset, data, _, _, _ = arxiv_dataloaders(
        adjacency='norm-lapl',
    )
    # Set Model
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = dataset.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = dataset.num_classes,
        dropout = model_kwargs['dropout'],
        phantom_grad = model_kwargs['phantom_grad'],
        beta_init = model_kwargs['beta_init'],
        gamma_init = model_kwargs['gamma_init'],
        tol = model_kwargs['tol'],
    )
    # Load Model
    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, data

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[data.train_mask], data.y.squeeze()[data.train_mask], reduction='mean')
    loss.backward()
    optimizer.step()

    evaluator = Evaluator('ogbn-arxiv')
    y_pred = output.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    
    return loss.item(), acc


@torch.no_grad()
def validate(cfg, data, model):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[data.val_mask], data.y.squeeze()[data.val_mask], reduction='mean')
    
    evaluator = Evaluator('ogbn-arxiv')
    y_pred = output.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']
    return loss.item(), acc


@torch.no_grad()
def test(cfg, data, model):
    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[data.test_mask], data.y.squeeze()[data.test_mask],reduction='mean')
    
    evaluator = Evaluator('ogbn-arxiv')
    y_pred = output.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']
    return loss.item(), acc


#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model):
    args = cfg.train
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'],eta_min = args['lr']/100)

    best = 1e8
    for epoch in range(args['epochs']):

        model.train()
        start = time.time()
        train_loss, train_acc = train(cfg, data, model, optimizer)
        scheduler.step()
        end = time.time()
        val_loss, val_acc = validate(cfg, data, model)

        perf_metric = val_loss #your performance metric here

        if perf_metric < best:
            best = perf_metric
            best_acc = val_acc
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
                },
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1
        # Log results
        wandb.log({'epoch':epoch,
            'fwd_itr':model.igl.itr,
            'train_loss':train_loss,
            'train_acc':train_acc,
            'val_loss':val_loss,
            'val_acc':val_acc,
            'best':best,
            'time':end-start})
        print(f'Epoch({epoch}) '
            f'| itr({model.igl.itr:03d}) '
            f'| train({100*train_acc:.2f},{train_loss:.4f}) '
            f'| val({100*val_acc:.2f},{val_loss:.4f}) '
            f'| best({best:.4f}) '
            f'| time({end-start:.4f})'
            )

        if bad_itr>args['patience']:
            break

    return best_acc

#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_folds(cfg):
    kfolds = 10 # we report the mean accuracy of 100 runs with random node ordering
    val_accs = [ None for _ in range(kfolds) ]
    test_accs = [ None for _ in range(kfolds) ]

    for k in range(kfolds):
        # Load
        model, data = load(cfg)
        model.to(cfg.setup['device'])
        data.to(cfg.setup['device'])

        total = sum(data.train_mask) + sum(data.val_mask) + sum(data.test_mask)
        print(f'Fold {k} Splits: train({100*sum(data.train_mask)/total:.2f})'
            f'\tval({100*sum(data.val_mask)/total:.2f})'
            f'\ttest({100*sum(data.test_mask)/total:.2f})'
            f'\ttrv({sum(data.train_mask)+sum(data.val_mask)})'
        )
        if cfg.setup['train']:
            val_acc = run_training(cfg, data, model)
            val_accs[k] = val_acc

        # Test
        test_loss, test_acc = test(cfg, data, model)
        test_accs[k] = test_acc
        
    print({'val_mean':np.mean(val_accs),
        'val_std':np.std(val_accs),
        'test_mean':np.mean(test_accs),
        'test_std':np.std(test_accs)})
    wandb.log({'val_mean':np.mean(val_accs),
        'val_std':np.std(val_accs),
        'test_mean':np.mean(test_accs),
        'test_std':np.std(test_accs)})

    return 1


#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/tasks/", config_name="arxiv")
def run_arxiv(cfg):
    # Initialize settings to wandb server
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(
        dir='/root/workspace/out/',
        entity='pr-inspired-aggregation',
        mode=mode,
        name='prgnn-arxiv-scheduled',
        project='pr-inspired-aggregation',
        tags=['arxiv', 'fixed'],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    run_folds(cfg)
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_arxiv()
