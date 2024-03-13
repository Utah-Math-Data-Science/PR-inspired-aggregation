#!/usr/bin/python3
import os, json, sys

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
import random
import torch
from torch.nn import LayerNorm, Linear, Module, Parameter, ReLU, Sequential, Tanh
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


from datasets.webkb_dataset import webkb_dataloaders
sys.path.append('/root/workspace/PR-inspired-aggregation/agg/')
from conv import ImplicitLayer
from functions import ReLU

"""
PyG TU Dataset
Splits: 80/10/10
"""

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------

class Model(MessagePassing):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: int,
        beta: float,
        gamma: float,
        max_iter: int,
        graph_size: int,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.act = ReLU()
        self.enc = Linear(in_channels, hidden_channels, bias=True)
        self.dec = Linear(hidden_channels, out_channels, bias=True)
        self.H = Linear(hidden_channels, graph_size, bias=False)
        self.f_theta = Linear(hidden_channels, hidden_channels, bias=True)
        self.max_iter = max_iter

        nonlin_module = ReLU()
        self.beta = Parameter(torch.tensor(beta,dtype=torch.float32))
        self.gamma = Parameter(torch.tensor(gamma,dtype=torch.float32))
        pass

    def forward(self, x, edge_index, edge_weight, u0=None):
        gamma = (1 + torch.abs(2*torch.sigmoid(self.beta)-1)) + self.act(self.gamma)
        edge_index, edge_weight = add_self_loops(edge_index, self.beta * edge_weight)
        inverse_matrix = self.invert_matrix(edge_index, edge_weight, maybe_num_nodes(edge_index))
        edge_index, edge_weight = self.dense_to_edge(inverse_matrix)
        x = self.enc(x.to(torch.float32))
        f = self.f_theta(x)
        #f = F.dropout(f, self.dropout, training=self.training)
        #for i in range(self.max_iter):
        h = self.propagate(edge_index, x=self.H.weight, edge_weight=edge_weight)
        x = self.act(2*self.H.weight - gamma*h - f)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act( x )
        x = self.dec(x)
        return x

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1,1) * x_j

    def invert_matrix(self, edge_index, edge_weight, num_nodes):
        dense_matrix = self.edge_to_dense(edge_index, edge_weight, num_nodes)
        # Compute inverse
        inverse_matrix = torch.inverse(dense_matrix)
        return inverse_matrix

    def edge_to_dense(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index
        dense_matrix = torch.zeros(num_nodes, num_nodes, device=self.beta.device)
        dense_matrix[row, col] = edge_weight
        return dense_matrix

    def dense_to_edge(self, matrix):
        row, col = matrix.nonzero(as_tuple=True)
        edge_weight = matrix[row, col]
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_weight

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

    dataset, data, data.train_mask, data.val_mask, data.test_mask = webkb_dataloaders(
        name=cfg.load['dataset'],
        split = cfg.load['split'],
        adjacency='symm-norm',
        beta = cfg.model['beta_init']
    )

    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = dataset.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = dataset.num_classes,
        dropout = model_kwargs['dropout'],
        beta = model_kwargs['beta_init'],
        gamma = model_kwargs['gamma_init'],
        max_iter = model_kwargs['max_iter'],
        graph_size = maybe_num_nodes(dataset.data.edge_index)
    )

    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, dataset

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    pred = output[data.train_mask].max(1)[1]
    acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    return loss.item(), acc


@torch.no_grad()
def validate(cfg, data, model):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    
    pred = output[data.val_mask].max(1)[1]
    acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    return loss.item(), acc


@torch.no_grad()
def test(cfg, data, model):
    model.eval()
    output = model(data.x,data.edge_index,data.edge_weight)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    
    pred = output[data.test_mask].max(1)[1]
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return loss.item(), acc

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model):
    args = cfg.train
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    best = 1e8
    bad_itr = 0
    for epoch in range(args['epochs']):
        train_loss, train_acc = train(cfg, data, model, optimizer)
        val_loss, val_acc = validate(cfg, data, model)

        perf_metric = 1-val_acc #your performance metric here

        if perf_metric < best:
            best = perf_metric
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc},
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1
        # Log results
        wandb.log({'epoch':epoch,
            'train_loss':train_loss,
            'val_loss':val_loss,
            'best':best})
        print('Epoch: {:03d}, '
            'train_loss: {:.7f}, '
            'train_acc: {:2.2f}, '
            'val_loss: {:.7f}, '
            'val_acc: {:2.2f}, '
            'perf_metric: {:2.2f}, '
            'best: {:2.2f}, '
            ''.format(epoch+1, train_loss, 100*train_acc, val_loss, val_acc*100, perf_metric, best))

        if bad_itr>args['patience']:
            break

    return best

#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_folds(cfg):
    kfolds = 10
    val_accs = [ None for _ in range(kfolds) ]
    test_accs = [ None for _ in range(kfolds) ]

    _, dataset = load(cfg)
    data = dataset[0]
    data.to(cfg.setup['device'])

    masks = [data.train_mask, data.val_mask, data.test_mask]

    original_path = cfg.load['checkpoint_path']

    for k in range(kfolds):
        cfg['load']['checkpoint_path']=original_path[:-3]+f'_fold_{k}.pt'
        # reset params
        model, _ = load(cfg)
        model.to(cfg.setup['device'])

        # Split Masks
        data.train_mask = index_to_mask(masks[0][:,k], data.num_nodes)
        data.val_mask = index_to_mask(masks[1][:,k], data.num_nodes)
        data.test_mask = index_to_mask(masks[2][:,k], data.num_nodes)
        total = sum(data.train_mask) + sum(data.val_mask) + sum(data.test_mask)

        print(f'Fold {k} Splits: train({100*sum(data.train_mask)/total:.2f})'
            f'\tval({100*sum(data.val_mask)/total:.2f})'
            f'\ttest({100*sum(data.test_mask)/total:.2f})'
            f'\ttrv({sum(data.train_mask)+sum(data.val_mask)})'
        )

        # Train
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


@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/tasks/", config_name="webkb")
def run_webkb(cfg):
    # Initialize settings to wandb server
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(
        dir='/root/workspace/out/',
        entity='utah-math-data-science',
        mode=mode,
        name='prgnn-'+cfg.load['dataset'],
        project='pr-inspired-aggregation',
        tags=['test', 'gcn-conv', cfg.load['dataset'], cfg.load['split']],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    run_folds(cfg)
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_webkb()
