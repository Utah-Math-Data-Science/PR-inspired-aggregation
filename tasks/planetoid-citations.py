#!/usr/bin/python3
import os
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
import torch
from torch.nn import GELU, Linear, Module, Sequential, Tanh
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything

from datasets.planetoid_citations import planetoid_citation_dataloaders
sys.path.append('/root/workspace/PR-inspired-aggregation/agg/')
from conv import ImplicitLayer
from functions import ReLU

"""
PyG Planetoid Dataset
    Split: 20 Train per class
    10-Fold Cross Validation
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
        max_iter: int,
        fix_params: bool,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.act = GELU()
        self.enc = Linear(in_channels, hidden_channels, bias=True)
        self.dec = Linear(hidden_channels, out_channels, bias=True)

        nonlin_module = ReLU()
        bias_module = Sequential(Linear(hidden_channels, 2*hidden_channels, bias=False), Tanh(), Linear(2*hidden_channels, hidden_channels, bias=False))
        # bias_module = Linear(hidden_channels, hidden_channels, bias=False)
        self.igl = ImplicitLayer(nonlin_module,
                                 bias_module,
                                 phantom_grad=phantom_grad,
                                 beta_init=beta_init,
                                 gamma_init=gamma_init,
                                 tol=tol,
                                 sigmoid=True,
                                 max_iter=max_iter,
                                 fix_params=fix_params,
                                 valid_u0=False)
        pass

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.enc(x)
        x = x + self.act( self.igl(x, edge_index, edge_weight) )
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.dec(x)
        return x

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

def load(cfg):
    args = cfg.load
    # Set Transforms
    transform = T.NormalizeFeatures()
    # Load Dataset
    dataset, _, _, _, _ = planetoid_citation_dataloaders(
        name=args['dataset'],
        adjacency='symm-norm',
        split='geom-gcn'
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
        max_iter = model_kwargs['max_iter'],
        tol = model_kwargs['tol'],
        fix_params = model_kwargs['fix_params'],
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
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    
    pred = output[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss.item(), acc


@torch.no_grad()
def validate(cfg, data, model, mask):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[mask], data.y[mask])
    
    pred = output[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss.item(), acc


@torch.no_grad()
def test(cfg, data, model, mask):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[mask], data.y[mask])
    
    pred = output[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss.item(), acc


#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model, tr_mask, val_mask):
    args = cfg.train
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])


    best = 1e8
    for epoch in range(args['epochs']):

        model.train()
        start = time.time()
        train_loss, train_acc = train(cfg, data, model, optimizer, tr_mask)
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
                'acc': val_acc,
                },
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1

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
            f'| beta({torch.abs(2*torch.sigmoid(model.igl.beta)-1):.3f}) '
            f'| gamma({model.igl.gamma:.3f}) '
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

    _, dataset = load(cfg)
    data = dataset[0]
    data.to(cfg.setup['device'])

    val_accs = [ None for _ in range(kfolds) ]
    test_accs = [ None for _ in range(kfolds) ]
    original_path = cfg.load['checkpoint_path']

    for k in range(kfolds):
        cfg['load']['checkpoint_path']=original_path[:-3]+f'_fold_{k}.pt'
        # Load
        model, _ = load(cfg)
        model.to(cfg.setup['device'])

        train_mask = data.train_mask[:,k]
        val_mask = data.val_mask[:,k]
        test_mask = data.test_mask[:,k]

        total = sum(train_mask) + sum(val_mask) + sum(test_mask)
        print(f'Fold {k} Splits: train({100*sum(train_mask)/total:.2f})'
            f'\tval({100*sum(val_mask)/total:.2f})'
            f'\ttest({100*sum(test_mask)/total:.2f})'
            f'\ttrv({sum(train_mask)+sum(val_mask)})'
        )

        val_acc = run_training(cfg, data, model, train_mask, val_mask)
        val_accs[k] = val_acc

        # Test
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(cfg.setup['device'])
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

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/tasks/", config_name="planetoid-citations")
def run_planetoid(cfg):
    # Initialize settings to wandb server
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(
        dir='/root/workspace/out/',
        entity='pr-inspired-aggregation',
        mode=mode,
        name='prgnn-'+cfg.load['dataset'],
        project='pyg-planetoid',
        tags=['prgnn', cfg.load['dataset'], 'geom-gcn'],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    run_folds(cfg)
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_planetoid()