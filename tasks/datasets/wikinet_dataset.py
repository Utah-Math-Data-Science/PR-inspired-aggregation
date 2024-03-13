"""
Wikipedia Network (WikiNet)

Web network of Wikipedia webpages
using a bag-of-words representation
for node-classification tasks.

This file is a loader for variations of the dataset.

"""

from typing import Optional

import torch

from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `name`: (str) name of dataset to use `chameleon`, `crocodile` or `squirrel`
    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is a connecting citation
        + `symm-norm`: edge indices by taking the symmetric norm of `base`
    `split`: (str) splits to use for learning
        + `geom-gcn`: the geom-gcn splits which contain 10 fold with 
        + `random-60/20/20`: randomly split with 60% training nodes 20% validation and 20% test nodes.
    #TODO: `folds`: (int) generate *n* random folds in randomly split data

"""

def wikinet_dataloaders(
    name : str = 'chameleon',
    adjacency : str = 'base',
    split : str = 'geom-gcn',
    folds : Optional[int] = None,
):

    name = name.lower()

    assert(name.lower() in ['chameleon','crocodile','squirrel']), f'Dataset not recognized: {name}'
    assert(adjacency in ['base','symm-norm']), f'Adjacency not recognized: {adjacency}'
    assert(split in ['geom-gcn','random-60/20/20']), f'Split not recognized: {split}'
    assert(split not in ['geom-gcn'] or name not in ['crocodile'] ), f'Split and name is incompatible: {split},{name}'

    split_kwargs = {}
    if split != 'geom-gcn':
        split_kwargs.update({'geom_gcn_preprocess':False})

    transform = [T.ToUndirected()]
    if adjacency=='symm-norm': transform.append(T.GCNNorm())

    dataset = WikipediaNetwork(root='/root/workspace/data/'+name,
        name = name,
        transform = T.Compose(transform),
        **split_kwargs
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


"""
Conversions
~~~~~~~~~~~

"""

classes = {
    'chameleon':5,
    'crocodile':5,
    'squirrel':5,
}

nodes = {
    'chameleon':2277,
    'crocodile':183,
    'squirrel':251,
}


"""
Test
~~~~~~~~~~~

"""

if __name__ == '__main__':
    dataset, data, tr_mask, val_mask, test_mask = wikinet_dataloaders(name='crocodile', split='random-60/20/20')
    print(dataset)
    print(dataset.num_classes, data.num_nodes)
    print(data)
    print(sum(tr_mask),sum(tr_mask)/len(tr_mask))
    print(sum(val_mask),sum(val_mask)/len(val_mask))
    print(sum(test_mask),sum(test_mask)/len(test_mask))
