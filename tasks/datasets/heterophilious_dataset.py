"""
Heterophilious Graph Datasets

A dataset with heterophilious graph data structure.

This file is a loader for variations of the dataset.

"""

from typing import Any, Optional

import torch
from torch.utils.data import random_split
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
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
def heterophilious_dataloaders(
    name : str = 'Roman-empire',
    adjacency : str = 'default',
    batch_size : int = 128,
):

    assert(name in dominant_targets), f'Only dominant targets are currently supported.\nUnrecognized target: {name}'
    assert(adjacency in ['default','sym-norm']), f'Adjacency not recognized: {adjacency}'

    transform = [] #[T.ToUndirected()]
    if adjacency=='sym-norm': transform.append(T.GCNNorm(add_self_loops=False))

    dataset = HeterophilousGraphDataset(
        root="/root/workspace/data/",
        name=name,
        transform=T.Compose(transform),
    )
    data = dataset[0]

    return dataset, data, data.train_mask, data.val_mask, data.test_mask 


"""
Data Splits
~~~~~~~~~~~~~~~

    Split the dataset containing into training, validation and test sets:

    `random_splits`: #TODO: Working on this for graph classification
    
"""
def fixed_splits(dataset):

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)

    return random_split(dataset, [num_training, num_val, num_test])
    # return dataset[train_idxs], dataset[val_idxs], dataset[test_idxs]

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset
#-----------------------------------------------------------------------------------------------------------------------------------------------------

import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected


class HeterophilousGraphDataset(InMemoryDataset):
    r"""The heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Roman-empire"`,
            :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"`,
            :obj:`"Questions"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Roman-empire
          - 22,662
          - 32,927
          - 300
          - 18
        * - Amazon-ratings
          - 24,492
          - 93,050
          - 300
          - 5
        * - Minesweeper
          - 10,000
          - 39,402
          - 7
          - 2
        * - Tolokers
          - 11,758
          - 519,000
          - 10
          - 2
        * - Questions
          - 48,921
          - 153,540
          - 301
          - 2
    """
    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        
        self.name = name.lower().replace('-', '_')
        assert self.name in [
            'roman_empire',
            'amazon_ratings',
            'minesweeper',
            'tolokers',
            'questions',
            'chameleon_filtered',
            'squirrel_filtered',
            'texas_4_classes',
        ]

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self):
        raw = np.load(self.raw_paths[0], 'r')
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

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