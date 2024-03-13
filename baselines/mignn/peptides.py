#!/usr/bin/python3
"""
PRGNN Peptides
==============

+ Define:     Model
+ Initialize: Config/Model/Dataset
+ Learning:   Train/Validate/Test
+ Drivers:    Main/Hydra/Fold/Train

"""
import os
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.nn import BatchNorm1d, Dropout, GELU, LayerNorm, Linear, Module, ReLU, Sequential, SiLU, Tanh, Transformer
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.graphgym import global_add_pool, global_mean_pool
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.seed import seed_everything
from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder

sys.path.append('/root/workspace/PR-inspired-aggregation/agg/')
from _conv import MonotoneImplicitGraph, CayleyLinear, ReLU, TanH
from _deq import ForwardBackward, ForwardBackwardAnderson, PeacemanRachford, PeacemanRachfordAnderson, PowerMethod
sys.path.append('/root/workspace/PR-inspired-aggregation/tasks/datasets/')
from peptides_dataset import peptide_dataloaders

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------

class Model(Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: int,
        tol: float,
        max_iter: int,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.act = ReLU()
        self.node_enc = AtomEncoder(hidden_channels)
        self.enc = Sequential(Linear(hidden_channels, hidden_channels, bias=True),
                              Dropout(dropout),
                              GELU(),
                              Linear(hidden_channels, hidden_channels, bias=True)
                            )
        self.dec = Sequential(Linear(hidden_channels, hidden_channels, bias=True),
                              Dropout(dropout),
                              GELU(),
                              Linear(hidden_channels, out_channels, bias=True)
                            )
        # self.dec = Linear(hidden_channels, out_channels, bias=True)

        self.norm = BatchNorm1d(hidden_channels)
        lin_module = CayleyLinear(hidden_channels, hidden_channels, None, invMethod='direct', adj=None, device='cuda')
        nonlin_module = ReLU()
        solver = ForwardBackwardAnderson(lin_module, nonlin_module, max_iter=max_iter, kappa=1.0, tol=tol)
        self.igl = MonotoneImplicitGraph(lin_module, nonlin_module, solver)
        pass

    def forward(self, data, x, edge_index, edge_weight, batch, u0=None):
        adj = torch.sparse_coo_tensor(edge_index, edge_weight).to('cuda')
        self.igl.lin_module.set_adj(adj,None)
        self.igl.solver.lin_module.set_adj(adj,None)
        data = self.node_enc(data)
        x = data.x
        x = self.act(x)
        x = self.enc(x)
        x = self.igl(x.T).t()
        x = self.norm(x)
        x = self.act(x)
        x = global_mean_pool(x, batch)
        x = self.dec(x)
        return x

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------------------------------------------------------

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)

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

    if args['sweep'] and args['train']:
        run_id = wandb.run.id
        cfg['load']['checkpoint_path']=cfg['load']['checkpoint_path'][:-3]+f'-ID({run_id}).pt'

    pass

#----------------------------------------------------------------------------------------------------------------------------------------------------

def load(cfg):
    args = cfg.load

    data, train_dl, val_dl, test_dl = peptide_dataloaders(
        aug_dim = cfg.model['hidden_channels'],
        adjacency = 'symm-norm',
        split = 'fixed_70/15/15',
        batch_size = args['batch_size'],
    )

    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = data.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = data.num_classes,
        dropout = model_kwargs['dropout'],
        tol = model_kwargs['tol'],
        max_iter = model_kwargs['max_iter'],
    )
    #print param count
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    # exit()

    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_dl, val_dl, test_dl

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data, x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch)
    loss = F.binary_cross_entropy_with_logits(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item(), output

@torch.no_grad()
def validate(cfg, data, model):
    model.eval()
    output = model(data, x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch)
    loss = F.binary_cross_entropy_with_logits(output, data.y)
    return loss.item(), output

@torch.no_grad()
def test(cfg, data, model):
    model.eval()
    output = model(data, x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, batch=data.batch)
    loss = F.binary_cross_entropy_with_logits(output, data.y)
    return loss.item(), output

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, model, train_dl, val_dl):
    args = cfg.train

    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5)

    model = model.to(cfg.setup['device'])

    best = 1e8
    for epoch in range(args['epochs']):

        model.train()
        train_loss, count = 0, 0
        start = time.time()

        pred_list, lable_list = [], []
        for i,data in enumerate(train_dl):
            data = data.to(cfg.setup['device'])
            batch_loss, preds = train(cfg, data, model, optimizer)

            pred_list.append(preds)
            lable_list.append(data.y)

            batch_size = data.y.shape[0]
            train_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Train({epoch}) '
                    f'| batch({i:03d})'
                    f'| loss({batch_loss:.4f})'
                    )

        end = time.time()
        train_ap = eval_ap(torch.cat(lable_list), torch.cat(pred_list))
        train_loss = train_loss/count
        scheduler.step(train_loss)
        
        model.eval()
        val_loss, count = 0, 0
        pred_list, lable_list = [], []
        for i,data in enumerate(val_dl): 
            data = data.to(cfg.setup['device'])
            batch_loss, preds = validate(cfg, data, model)

            pred_list.append(preds)
            lable_list.append(data.y)

            batch_size = data.y.shape[0]
            val_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Valid({epoch}) | batch({i:03d}) | loss({batch_loss:.4f})')

        val_loss = val_loss/count
        val_ap = eval_ap(torch.cat(lable_list), torch.cat(pred_list))
        perf_metric = 1-val_ap #your performance metric here
        lr = optimizer.param_groups[0]['lr']

        if perf_metric < best:
            best = perf_metric
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': lr,
                'loss': val_loss,
                },
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1

        wandb.log({'epoch':epoch,
            'train_loss':train_loss,
            'train_ap':train_ap,
            'val_loss':val_loss,
            'val_ap':val_ap,
            'best':best,
            'lr':lr,
            'time':end-start})
        print(f'Epoch({epoch}) '
            f'| train({train_loss:.4f},{train_ap:.4f}) '
            f'| val({val_loss:.4f},{val_ap:.4f}) '
            f'| lr({lr:.2e}) '
            f'| best({best:.4f}) '
            f'| time({end-start:.4f})'
            f'\n')

        if bad_itr>args['patience']:
            break

    return best

#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/tasks/", config_name="peptides")
def run_qm9(cfg):
    """
    Execute run saving details to wandb server.
    """
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(entity='utah-math-data-science',
                project='pr-inspired-aggregation',
                mode=mode,
                name='prgnn-peptides',
                dir='/root/workspace/out/',
                tags=['peptides', 'prgnn'],
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, train_dl, val_dl, test_dl = load(cfg)
    print(model)

    if cfg.setup['train']:
        run_training(cfg, model, train_dl, val_dl)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.setup['device'])

    test_loss, count = 0, 0
    pred_list, lable_list = [], []

    for data in test_dl:
        data.to(cfg.setup['device'])
        batch_loss, preds = test(cfg, data, model)

        pred_list.append(preds)
        lable_list.append(data.y)

        batch_size = data.y.shape[0]
        test_loss += batch_loss * batch_size
        count += batch_size
    test_loss = test_loss/count
    test_ap = eval_ap(torch.cat(lable_list), torch.cat(pred_list))

    print(f'\ntest({test_loss},{test_ap:.4f})')
    wandb.log({'test_loss':test_loss, 'test_ap':test_ap})
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_qm9()
