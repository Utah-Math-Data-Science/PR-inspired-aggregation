"""
PPI Graph Dataset

A dataset with heterophilious graph data structure.

This file is a loader for variations of the dataset.

"""
from typing import Any, Optional

import torch
from torch.utils.data import random_split
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `name`: (str) label target for training
    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `default`: edge indices indicate default connection
        + `sym-norm`: symmetric normalization of default graph
    `batch_size`: (int) maximum batch size for graphs

"""
def ppi_dataloaders(
    aug_dim: int,
    adjacency : str = 'default',
    batch_size : int = 128,
):

    assert(adjacency in ['default','sym-norm']), f'Adjacency not recognized: {adjacency}'

    transform = [AugmentU0(dim=aug_dim), T.ToUndirected()]#, T.NormalizeFeatures()]
    if adjacency=='sym-norm': transform.append(T.GCNNorm(add_self_loops=False))

    train_dataset = PPI(
        root="/root/workspace/data/",
        split='train',
        transform=T.Compose(transform),
    )
    val_dataset = PPI(
        root="/root/workspace/data/",
        split='train',
        transform=T.Compose(transform),
    )
    test_dataset = PPI(
        root="/root/workspace/data/",
        split='train',
        transform=T.Compose(transform),
    )

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_dl, val_dl, test_dl


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Transforms
#-----------------------------------------------------------------------------------------------------------------------------------------------------

class AugmentU0(T.BaseTransform):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    def __call__(self, data: Any) -> Any:
        data.u0 = torch.randn((*data.x.shape[:-1], self.dim))
        return data

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Look-Up Tables
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dictionaries
~~~~~~~~~~~~~

"""

dominant_targets = [
    'Roman-empire',
    'Amazon-ratings',
    'Minesweeper',
    'Tolokers',
    'Questions',
    'Chameleon-filtered',
    'Squirrel-filtered',
    'Texas-4-classes',
]


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    dataset, _, _, _, _ = heterophilious_dataloaders()
    print(dataset[0])
    data = dataset[0]
    total = sum(data.train_mask) + sum(data.val_mask) + sum(data.test_mask)
    print(sum(data.train_mask)/total, sum(data.val_mask)/total, sum(data.test_mask)/total)