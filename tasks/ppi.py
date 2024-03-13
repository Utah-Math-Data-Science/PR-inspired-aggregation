#!/usr/bin/python3
import os
import random
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import Linear, Module, ReLU
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything

from datasets.ppi_dataset import ppi_dataloaders
sys.path.append('/root/workspace/PR-inspired-aggregation/agg/')
from conv import ImplicitLayer
from functions import ReLU

"""
PyG PPI Dataset
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
        self.act = ReLU()
        self.enc = Linear(in_channels, hidden_channels, bias=False)
        self.dec = Linear(hidden_channels, out_channels, bias=False)

        nonlin_module = ReLU()
        bias_module = Linear(hidden_channels, hidden_channels, bias=False)
        self.igl = ImplicitLayer(nonlin_module, bias_module, phantom_grad=phantom_grad, beta_init=beta_init, gamma_init=gamma_init, tol=tol)
        pass

    def forward(self, x, edge_index, edge_weight, batch, u0=None):
        x = self.enc(x.to(torch.float32))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act( self.igl(x, edge_index, edge_weight, u0=u0) )
        x = self.dec(x)
        return x, self.igl.u0
#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------------------------------------------------------

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Config/Model/Dataset
#----------------------------------------------------------------------------------------------------------------------------------------------------

def setup(cfg):
    args = cfg.setup

    seed_everything(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg['setup']['device'] = args['device'] if torch.cuda.is_available() else 'cpu'

    os.environ["WANDB_DIR"] = os.path.abspath(args['wandb_dir'])

    if args['sweep']:
        run_id = wandb.run.id
        cfg['load']['checkpoint_path']=cfg['load']['checkpoint_path'][:-3]+f'-ID({run_id}).pt'

    pass

#----------------------------------------------------------------------------------------------------------------------------------------------------

def load(cfg):
    args = cfg.load

    dataset, _, _, train_loader, val_loader, test_loader = ppi_dataloaders(
        aug_dim = cfg.model['hidden_channels'],
        adjacency='sym-norm',
        batch_size=args['batch_size'],
    )

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

    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_loader, val_loader, test_loader

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output, u0 = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch, u0=data.u0)
    data.u0 = u0
    loss = F.binary_cross_entropy_with_logits(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item(), output

@torch.no_grad()
def validate(cfg, data, model):
    model.eval()
    output, u0 = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch, u0=data.u0)
    loss = F.binary_cross_entropy_with_logits(output, data.y)
    return loss.item(), output

@torch.no_grad()
def test(cfg, data, model):
    model.eval()
    output, u0 = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch, u0=data.u0)
    loss = F.binary_cross_entropy_with_logits(output, data.y)
    return loss.item(), output

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, model, train_dl, val_dl):
    args = cfg.train

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'], amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)

    best = 1e8
    bad_itr = 0
    for epoch in range(args['epochs']):

        model.train()
        train_loss, train_pred, train_true = 0, [], []
        start = time.time()
        for data in train_dl:
            data = data.to(cfg.setup['device'])
            train_loss_, train_pred_ = train(cfg, data, model, optimizer)

            train_pred.append((train_pred_ > 0).float().cpu())
            train_true.append((data.y > 0).float().cpu())
            train_loss += train_loss_
        end = time.time()

        train_true, train_pred = torch.cat(train_true, dim=0).numpy(), torch.cat(train_pred, dim=0).numpy()
        train_acc = f1_score(train_true, train_pred, average='micro') if train_pred.sum() > 0 else 0
        
        model.eval()
        val_loss, val_pred, val_true = 0, [], []
        for i,data in enumerate(val_dl): 
            data = data.to(cfg.setup['device'])
            val_loss_, val_pred_ = validate(cfg, data, model)

            val_pred.append((val_pred_ > 0).float().cpu())
            val_true.append((data.y > 0).float().cpu())
            val_loss += val_loss_

        val_true, val_pred = torch.cat(val_true, dim=0).numpy(), torch.cat(val_pred, dim=0).numpy()
        val_acc = f1_score(val_true, val_pred, average='micro') if val_pred.sum() > 0 else 0
        
        train_loss, val_loss = train_loss/len(train_dl.dataset), val_loss/len(val_dl.dataset)

        perf_metric = 1-val_acc #your performance metric here

        if perf_metric < best:
            best = perf_metric
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc,
                'best': best},
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1

        wandb.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'train_acc':train_acc,
            'val_loss':val_loss,
            'val_acc':val_acc,
            'best':best,
            'time':end-start,
        })
        print(f'Epoch({epoch:04d}) '
            f'| train({train_loss:.4f},{train_acc:.2f}) '
            f'| val({val_loss:.4f},{val_acc:.2f}) '
            f'| best({best:.4f}) '
            f'| time({end-start:.4f})'
        )

        if bad_itr>args['patience']:
            break

    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/tasks/", config_name="ppi")
def run_ppi(cfg):
    """
    Execute run saving details to wandb server.
    """
    # Setup Weights and Bias
    wandb.init(
        dir='/root/workspace/out/',
        entity='utah-math-data-science',
        mode='disabled',
        name='prgnn-'+cfg.load['dataset'],
        project='pr-inspired-aggregation',
        tags=['prgnn', cfg.load['dataset']],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, train_dl, val_dl, test_dl = load(cfg)
    model = model.to(cfg.setup['device'])
    print(model)
    if cfg.setup['train']:
        run_training(cfg, model, train_dl, val_dl)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(cfg.setup['device'])
    test_loss, test_pred, test_true = 0, [], []
    for data in test_dl:
        data = data.to(cfg.setup['device'])
        test_loss_, test_pred_ = test(cfg, data, model)
        test_pred.append((test_pred_ > 0).float().cpu())
        test_true.append((data.y > 0).float().cpu())
        test_loss += test_loss_

    test_true, test_pred = torch.cat(test_true, dim=0).numpy(), torch.cat(test_pred, dim=0).numpy()
    test_acc = f1_score(test_true, test_pred, average='micro') if test_pred.sum() > 0 else 0
    test_loss = test_loss/len(test_dl)

    print(f'test ({test_loss:.4f},{test_acc:.4f})')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_ppi()
