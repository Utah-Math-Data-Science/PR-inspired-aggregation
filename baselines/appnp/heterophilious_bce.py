#!/usr/bin/python3
import json
import os
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
import random
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import GELU, LayerNorm, Linear, Module, ModuleList, Sequential, Tanh, ReLU
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T




from torch_geometric.nn import APPNP
sys.path.append('/root/workspace/PR-inspired-aggregation/tasks/datasets/')
from heterophilious_dataset import heterophilious_dataloaders

"""
Hetereophilious Dataset
    Split: 20 Train per class
    10-Fold Cross Validation
"""

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------

class Model(Module):
    def __init__(self, in_channels, hidden_channels, hidden_layers, out_channels, dropout, alpha, K):
        super(Model, self).__init__()
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.act = GELU()
        self.norm = LayerNorm(hidden_channels)
        self.enc = Linear(in_channels, hidden_channels, bias=False)
        self.dec = Linear(hidden_channels, out_channels, bias=False)
        self.conv = ModuleList([])
        for i in range(hidden_layers):
            self.conv.append(APPNP(K=K, alpha=alpha, normalize=False))

    def forward(self, x, edge_index, edge_weight):
        x = self.enc(x)
        x = self.norm(x)
        x = self.act(x)
        _layers = [x]
        for conv in self.conv:
            x = self.act( conv(x, edge_index=edge_index, edge_weight=edge_weight) )
            x = self.norm(x)
            _layers.append(x)
        x = self.dec(x)
        return x.squeeze()

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
    # Load Dataset
    assert args['dataset'] in ['Minesweeper', 'Tolokers', 'Questions'], 'Dataset is not a BCE dataset please use regular heterophilious.py'
    dataset, _, _, _, _ = heterophilious_dataloaders(
        name=args['dataset'],
        adjacency='sym-norm',

    )
    # Set Model
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = dataset.num_features,
        out_channels = 1,
        hidden_channels = model_kwargs['hidden_channels'],
        hidden_layers = model_kwargs['hidden_layers'],
        dropout = model_kwargs['dropout'],
        alpha = model_kwargs['alpha'],
        K = model_kwargs['K'],
    )
    # Load Model
    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, dataset

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer, mask):
    model.train()
    optimizer.zero_grad()
    output = model(data.x,data.edge_index,data.edge_weight)
    # output = F.log_softmax(output, dim=1)
    loss = F.binary_cross_entropy_with_logits(output[mask], data.y[mask].to(torch.float))
    loss.backward()
    optimizer.step()
    
    acc = roc_auc_score(data.y[mask].squeeze().detach().cpu().numpy(), output[mask].squeeze().detach().cpu().numpy())
    return loss.item(), acc


@torch.no_grad()
def validate(cfg, data, model, mask):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    # output = F.log_softmax(output, dim=1)
    loss = F.binary_cross_entropy_with_logits(output[mask], data.y[mask].to(torch.float))
    
    acc = roc_auc_score(data.y[mask].squeeze().detach().cpu().numpy(), output[mask].squeeze().detach().cpu().numpy())
    return loss.item(), acc


@torch.no_grad()
def test(cfg, data, model, mask):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    # output = F.log_softmax(output, dim=1)
    loss = F.binary_cross_entropy_with_logits(output[mask], data.y[mask].to(torch.float))
    
    acc = roc_auc_score(data.y[mask].squeeze().detach().cpu().numpy(), output[mask].squeeze().detach().cpu().numpy())
    return loss.item(), acc
#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model, tr_mask, val_mask):
    args = cfg.train
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduelr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)


    best = 1e8
    for epoch in range(args['epochs']):

        model.train()
        start = time.time()
        train_loss, train_acc = train(cfg, data, model, optimizer, tr_mask)
        scheduelr.step()
        end = time.time()
        val_loss, val_acc = validate(cfg, data, model, val_mask)

        perf_metric = (1-val_acc) #your performance metric here

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
            'train_loss':train_loss,
            'train_acc':train_acc,
            'val_loss':val_loss,
            'val_acc':val_acc,
            'best':best,
            'time':end-start})
        print(f'Epoch({epoch}) '
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

    original_path = cfg.load['checkpoint_path']

    for k in range(kfolds):
        cfg['load']['checkpoint_path']=original_path[:-3]+f'_fold_{k}.pt'
        # Load
        model, dataset = load(cfg)
        model.to(cfg.setup['device'])
        data = dataset[0]
        data.to(cfg.setup['device'])

        train_mask = data.train_mask[:,k]
        val_mask = data.val_mask[:,k]
        test_mask = data.test_mask[:,k]

        total = sum(train_mask) + sum(val_mask) + sum(test_mask)
        print(f'Fold {k} Splits: train({100*sum(train_mask)/total:.2f})'
            f'\tval({100*sum(val_mask)/total:.2f})'
            f'\ttest({100*sum(test_mask)/total:.2f})'
            f'\ttrv({sum(train_mask)+sum(val_mask)})'
        )
        if cfg.setup['train']:
            val_acc = run_training(cfg, data, model, train_mask, val_mask)
            val_accs[k] = val_acc

        # Test
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc = test(cfg, data, model, test_mask)
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

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/baselines/appnp", config_name="heterophilious")
def run_heterophilious(cfg):
    # Initialize settings to wandb server
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(
        dir='/root/workspace/out/',
        entity='utah-math-data-science',
        mode=mode,
        name='appnp-'+cfg.load['dataset'],
        project='pr-inspired-aggregation',
        tags=['appnp', cfg.load['dataset']],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    run_folds(cfg)
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_heterophilious()