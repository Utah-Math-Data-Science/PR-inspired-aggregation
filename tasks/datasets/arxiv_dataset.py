"""
arXiv

A citation network of the arXiv repository over
five domains: Computer Science, Physics, Mathematics,
Statistics, and Electrical Engineering. Used for
*large-scale* node-classification tasks.

This file is a loader for variations of the dataset.

"""

from typing import Optional
from torch_geometric.data import Data

import torch

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is a connecting citation
        + `symm-norm`: edge indices by taking the symmetric norm of `base`
    `split`: (str) splits to use for learning
        + `fixed`: use the internal data splitting function
    #TODO: `folds`: (int) generate *n* random folds in randomly split data

"""

def arxiv_dataloaders(
    adjacency : str = 'base',
    split : str = 'fixed',
    folds : Optional[int] = None,
):

    assert(adjacency in ['base','symm-norm', 'norm-lapl']), f'Adjacency not recognized: {adjacency}'
    assert(split in ['fixed']), f'Split not recognized: {split}'

    split_vals = list(split.split('-'))
    split_vals.extend([None]*(3-len(split_vals)))

    transform = []
    if adjacency=='symm-norm': transform.append(T.GCNNorm())
    elif adjacency=='norm-lapl': transform.append(NormLaplacian())

    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv',
        root='/root/workspace/data/',
        transform=T.Compose(transform),
    )

    if split=='fixed':
        data, train_mask, val_mask, test_mask = fixed_splits(dataset)

    return dataset, data, train_mask, val_mask, test_mask


"""
Splits
~~~~~~~

"""

def fixed_splits(dataset):
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    split_masks = {}
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f"{split}_mask"] = mask
        split_masks[f"{split}"] = data[f"{split}_mask"]
    
    data.train_mask = split_masks['train']
    data.val_mask = split_masks['valid']
    data.test_mask = split_masks['test']

    return data, split_masks['train'], split_masks['valid'], split_masks['test']


class NormLaplacian(T.BaseTransform):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (functional name: :obj:`gcn_norm`).

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """
    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def __call__(self, data: Data) -> Data:
        # data = T.ToUndirected()(data)
        data = T.GCNNorm()(data)
        if data.edge_weight is None:
            data.edge_weight = -torch.ones(data.edge_index.shape[1])
        else:
            data.edge_weight = - data.edge_weight
        data = T.AddSelfLoops()(data)
        return data


"""
Conversions
~~~~~~~~~~~

"""


"""
Test
~~~~~~~~~~~

"""

if __name__ == '__main__':
    dataset, data, tr_mask, val_mask, test_mask = arxiv_dataloaders()
    print(dataset)
    print(data)
    print(data.y.shape)
    print(sum(tr_mask),sum(tr_mask)/len(tr_mask))
    print(sum(val_mask),sum(val_mask)/len(val_mask))
    print(sum(test_mask),sum(test_mask)/len(test_mask))