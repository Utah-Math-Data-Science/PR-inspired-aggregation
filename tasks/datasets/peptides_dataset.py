"""
Peptides Functional 

The goal is use the molecular representation of peptides instead
of amino acid sequence representation ('peptide_seq' field in the file,
provided for possible baseline benchmarking but not used here) to test
GNNs' representation capability.

This file is a loader for variations of the dataset.

"""
from typing import Any, Optional

import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url
from torch_geometric.data import DataLoader, InMemoryDataset
import torch_geometric.transforms as T
from tqdm import tqdm

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is an existing bond
        + `radial`: edge indices filled if atoms are within radial cutoff
        + `full`: fully connected edge indices for each molecule
    `o3_attr`: (bool) allow use spherical harmonic edge featurization
    `lmax_attr`: (int) maximum for `o3_attr`
    `split`: (str) data splits for common ML papers
        + `fixed`: random split of 110_000 training, 1_000 validation and rest test
    `batch_size`: (int) maximum batch size for graphs

    The molecular properties are also normalized using the mean and
    mean average deviation. These are kept in dataset.stats and can
    be used to recompose the desired features.

"""
def peptide_dataloaders(
    aug_dim: int = 512,
    adjacency : Optional[str] = None,
    split : str = 'fixed_70/15/15',
    batch_size: int = 128,
):
    assert(adjacency in ['symm-norm', None]), f'Adjacency not recognized: {adjacency}'
    assert(split in ['fixed_70/15/15']), f'Split not recognized: {split}'
    pretransform = [AugmentU0(dim=aug_dim), T.ToUndirected(), T.GCNNorm()]
    # transform = [T.ToUndirected()]

    # if adjacency=='symm-norm': transform.append(T.GCNNorm())

    dataset = PeptidesFunctionalDataset(root='/root/workspace/data/',
        pre_transform = T.Compose(pretransform),
        # transform = T.Compose(transform)
    )
    if split=='fixed_70/15/15':
        s_dict = dataset.get_idx_split()
        dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
        dataset.train_mask = s_dict['train']
        dataset.val_mask = s_dict['val']
        dataset.test_mask = s_dict['test']
    
    train_loader = DataLoader(dataset[dataset.train_mask], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[dataset.val_mask], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.test_mask], batch_size=batch_size, shuffle=False)

    # !JB: There is an option for positional encodings that may be examined for future work.

    return dataset, train_loader, val_loader, test_loader

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset
#-----------------------------------------------------------------------------------------------------------------------------------------------------


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(self, root='/root/workspace/data/', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-functional')

        self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'
        self.version = '701eb743e899f4d793f0e13c8fa5a1b4'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_multi_class_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_multi_class_dataset.csv.gz'))
        smiles_list = data_df['smiles']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([eval(data_df['labels'].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict



"""
Transforms
~~~~~~~~~~

"""
class AugmentU0(T.BaseTransform):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    def __call__(self, data: Any) -> Any:
        data.u0 = torch.randn((data.x.shape[0], self.dim))
        return data




#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    dataset, train_dl, _, _ = peptide_dataloaders(aug_dim=0)
    print(dataset)
    print(dataset[0])
    for batch in train_dl:
        print(batch)
    # print(dataset.data.edge_index)
    # print(dataset.data.edge_index.shape)
    # print(dataset.data.x.shape)
    # print(dataset[100])
    # print(dataset[100].y)
    # print(dataset.get_idx_split())
    # total = dataset.get_idx_split()['train'].shape[0] + dataset.get_idx_split()['val'].shape[0] + dataset.get_idx_split()['test'].shape[0]
    # print(dataset.get_idx_split()['train'].shape[0]/total)
    # print(dataset.get_idx_split()['val'].shape[0]/total)
    # print(dataset.get_idx_split()['test'].shape[0]/total)