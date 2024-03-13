"""
Web Knowledge Base (WebKB)

Web network of Computer Science department
webpages using a bag-of-words representation
for node-classification tasks.

This file is a loader for variations of the dataset.

"""
import os

from typing import Optional

import torch

from torch_geometric.datasets import WebKB
import torch_geometric.transforms as T

import numpy as np
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix, to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data
import torch_geometric.transforms as T
from torch_sparse import coalesce

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `name`: (str) name of dataset to use `cornell`, `texas` or `wisconsin`
    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is a connecting citation
        + `symm-norm`: edge indices by taking the symmetric norm of `base`
    `split`: (str) splits to use for learning
        + `geom-gcn`: the geom-gcn splits which contain 10 fold with 
        + `random-60/20/20`: randomly split with 60% training nodes 20% validation and 20% test nodes.
    #TODO: `folds`: (int) generate *n* random folds in randomly split data

"""

def webkb_dataloaders(
    name : str = 'cornell',
    adjacency : str = 'base',
    split : str = 'geom-gcn',
    folds : Optional[int] = None,
    beta : Optional[float] = None,
):

    name = name.lower()

    assert(name.lower() in ['cornell','texas','wisconsin']), f'Dataset not recognized: {name}'
    assert(adjacency in ['base','symm-norm', 'inv']), f'Adjacency not recognized: {adjacency}'
    assert(split in ['geom-gcn','random-60/20/20']), f'Split not recognized: {split}'

    transform = [T.NormalizeFeatures(), T.ToUndirected()]
    if adjacency=='symm-norm': transform.append(T.GCNNorm())
    elif adjacency=='inv':
        self_loops = name=='wisconsin'
        transform.append(T.GCNNorm(add_self_loops=self_loops))
        transform.append(InverseMatrixTransform(beta))

    dataset = WebKB_new(root='/root/workspace/data/'+name+'_new',
        name = name,
        transform = T.Compose(transform),
    )
    data = dataset[0]

    if split=='random-60/20/20':
        data = random_splitter(data) 

    return dataset, data, data.train_mask, data.val_mask, data.test_mask 


"""
Splits
~~~~~~~~~~~

"""

def random_splitter(data):
    n = data.y.shape[0]
    tr_lb = int(.6*n)
    val_lb = int(.2*n)

    index = torch.randperm(n)

    train_idx = [i for i in range(n) if i in index[:tr_lb]]
    val_idx = [i for i in range(n) if i in index[tr_lb:tr_lb+val_lb]]
    test_idx = [i for i in range(n) if i in index[tr_lb+val_lb:]]

    def get_mask(idx):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

from torch_geometric.data import InMemoryDataset, download_url, Data
class WebKB_new(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'wisconsin']

        super(WebKB_new, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [
            '{}_split_0.6_0.2_{}.npz'.format(self.name, i) for i in range(10)
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.float)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)
        data = Data(x=x, edge_index=edge_index, y=y.to(torch.int64), train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
"""
Transforms
~~~~~~~~~~~

"""
class InverseMatrixTransform(T.BaseTransform):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, data):
        # Check if the attributes exist
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'edge_weight')

        num_nodes = data.num_nodes
        edge_index, edge_weight = add_self_loops(data.edge_index, self.beta * data.edge_weight, num_nodes=data.num_nodes)
        inverse_matrix = self.invert_matrix(edge_index, edge_weight, num_nodes)

        edge_index, edge_weight = self.dense_to_edge(inverse_matrix)

        data.edge_index = edge_index
        data.edge_weight = edge_weight
        return data

    def invert_matrix(self, edge_index, edge_weight, num_nodes):
        dense_matrix = self.edge_to_dense(edge_index, edge_weight, num_nodes)
        # Compute inverse
        inverse_matrix = torch.inverse(dense_matrix)
        return inverse_matrix

    def edge_to_dense(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index
        dense_matrix = torch.zeros(num_nodes, num_nodes)
        dense_matrix[row, col] = edge_weight
        return dense_matrix

    def dense_to_edge(self, matrix):
        row, col = matrix.nonzero(as_tuple=True)
        edge_weight = matrix[row, col]
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_weight

"""
Conversions
~~~~~~~~~~~

"""

classes = {
    'cornell':5,
    'texas':5,
    'wisconsin':5,
}

nodes = {
    'cornell':183,
    'texas':183,
    'wisconsin':251,
}


"""
Test
~~~~~~~~~~~

"""

if __name__ == '__main__':
    dataset, data, tr_mask, val_mask, test_mask = webkb_dataloaders(split='random-60/20/20')
    print(dataset)
    print(data)
    print(sum(tr_mask),sum(tr_mask)/len(tr_mask))
    print(sum(val_mask),sum(val_mask)/len(val_mask))
    print(sum(test_mask),sum(test_mask)/len(test_mask))
