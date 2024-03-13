#!/usr/bin/python3
"""
Synthetic Chains Dataset
    Split: 5/10/85

Adapted from 'EIGNN <https://github.com/liu-jc/EIGNN/tree/main>'_
Optimizer: Adam
Loss: NLL

hidden_channels: 16
num_eigenvectors: 100
gamma: 0.8
"""
import os
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix

from _conv import IDM_SGC
sys.path.append('/root/workspace/PR-inspired-aggregation/tasks/datasets/')
from synthetic import synth_chains_data

"""
Synthetic Chains Dataset
    Split: 5/10/85
"""

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------

class Model(Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma, dropout):
        super(Model, self).__init__()
        self.dropout = dropout
        start = time.time()
        self.EIGNN = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        end = time.time()
        wandb.log({'time', end-start})
        self.B = Linear(m, m_y, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()

    def forward(self, X):
        output = self.EIGNN(X.T).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.B(output)
        return output


#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------------------------------------------------------

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

#----------------------------------------------------------------------------------------------------------------------------------------------------

def memory_usage(device,key=''):
    memory_stats = {}
    memory_stats.update({f'{key}Allocated:': round(torch.cuda.memory_allocated(device)/1024**2,1)})
    memory_stats.update({f'{key}MaxAllocated:': round(torch.cuda.max_memory_allocated(device)/1024**2,1)})
    memory_stats.update({f'{key}Reserved:': round(torch.cuda.memory_reserved(device)/1024**2,1)})
    memory_stats.update({f'{key}MaxReserved:': round(torch.cuda.max_memory_reserved(device)/1024**2,1)})
    return memory_stats

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
    start = time.time()
    model = Model(
        adj=to_dense_adj(data.edge_index, edge_attr=data.edge_weight)[0].to(cfg.setup['device']),
        sp_adj=to_scipy_sparse_matrix(data.edge_index, edge_attr=data.edge_weight),
        m = data.num_features,
        m_y = data.num_classes,
        num_eigenvec=cfg.model['num_eigenvec'],
        gamma=cfg.model['gamma'],
        dropout=cfg.model['dropout'],
    )
    end = time.time()
    print(f'MODEL INITIALIZATION TIME: {end-start}')
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
    output = model(data.x)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    return loss.item(), acc

def validate(cfg, data, model):
    model.eval()
    output = model(data.x)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    
    acc = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss.item(), acc


def test(cfg, data, model):
    model.eval()
    output = model(data.x)
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    
    acc = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss.item(), acc

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model):
    args = cfg.train
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    best = 1e8
    best_loss = 1e5
    bad_itr = 0

    for epoch in range(args['epochs']):
        pretrain_mem = memory_usage(cfg.setup['device'],key='Pretrain')
        start = time.time()
        train_loss, train_acc = train(cfg, data, model, optimizer)
        end = time.time()
        postrain_mem = memory_usage(cfg.setup['device'],key='Postrain')
        val_loss, val_acc = validate(cfg, data, model)
        posval_mem = memory_usage(cfg.setup['device'],key='Postval')

        perf_metric = 1-val_acc

        if perf_metric<best:
            best = perf_metric
            best_loss = min(val_loss, best_loss)
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
        log = {'epoch':epoch,
            'train_loss':train_loss,
            'val_loss':val_loss,
            'perf_metric':perf_metric,
            'best':best,
            'time':end-start,}
        log.update(pretrain_mem)
        log.update(postrain_mem)
        log.update(posval_mem)
        wandb.log(log)
        print(
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

@hydra.main(version_base=None, config_path="/root/workspace/PR-inspired-aggregation/baselines/eignn/", config_name="synth_chains.yaml")
def run_chains(cfg):
    # Initialize settings to wandb server
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(entity='utah-math-data-science',
                project='pr-inspired-aggregation',
                mode=mode,
                name='eignn-memory-timing-B'+str(cfg.data['chain_len']),
                dir='/root/workspace/out/',
                tags=['synth-chains', 'ignn', str(cfg.data['chain_len'])],
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    wandb.define_metric('time', summary='mean')
    
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))

    model, data= load(cfg)
    print(model)
    model.to(cfg.setup['device'])
    data.to(cfg.setup['device'])

    if cfg.setup['train']:
        run_training(cfg, data, model)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.setup['device'])
    test_loss, test_acc = test(cfg, data, model)
    print(f'Test Accuracy: {100*test_acc}')
    wandb.log({'test_loss':test_loss, 'test_acc':test_acc})
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_chains()