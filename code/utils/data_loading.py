import os

from rdkit import Chem

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as GeometricData, DataLoader as GeometricDataLoader
from sklearn.preprocessing import StandardScaler

from typing import Union, List, Tuple, Any, Optional

from .chemprop_featurisation import atom_features, bond_features, get_atom_constants


def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)
    return (Chem.MolToSmiles(mol))


class GraphMoleculeDataset(TorchDataset):
    def __init__(self, csv_path: str, max_atomic_num: int, smiles_column_name: str, label_column_name: Union[str, List[str]],
                 scaler: Optional[StandardScaler] = None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.atom_constants = get_atom_constants(max_atomic_num)
        self.num_atom_features = sum(len(choices) for choices in self.atom_constants.values()) + 2
        self.num_bond_features = 13
        self.scaler = scaler
        self.label_dims = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: Union[torch.Tensor, slice, List]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            slice_step = idx.step if idx.step else 1
            idx = list(range(idx.start, idx.stop, slice_step))
        if not isinstance(idx, list):
            idx = [idx]

        selected = self.df.iloc[idx]
        smiles = selected[self.smiles_column_name].values
        self.label_dims = selected[self.label_column_name].values.ndim
        if self.scaler:
            if selected[self.label_column_name].values.ndim == 1:
                labels = torch.Tensor(self.scaler.transform(np.expand_dims(selected[self.label_column_name].values, axis=1)))
            else:
                labels = torch.Tensor(self.scaler.transform(selected[self.label_column_name].values))
        else:
            labels = torch.Tensor(selected[self.label_column_name].values)
        smiles = [remove_smiles_stereo(s) for s in smiles]
        rdkit_mols = [Chem.MolFromSmiles(s) for s in smiles]

        atom_feat = [torch.Tensor([atom_features(atom, self.atom_constants) for atom in mol.GetAtoms()]) for mol in rdkit_mols]
        bond_feat = [torch.Tensor([bond_features(bond) for bond in mol.GetBonds()]) for mol in rdkit_mols]
        edge_index = [torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T for mol in rdkit_mols]

        geometric_data_points = [GeometricData(x=atom_feat[i], edge_attr=bond_feat[i], edge_index=edge_index[i], y=labels[i]) for i in range(len(atom_feat))]
        for i, data_point in enumerate(geometric_data_points):
            # data_point.rdkit_mol = rdkit_mols[i]
            data_point.smiles = smiles[i]

        if len(geometric_data_points) == 1:
            return geometric_data_points[0]
        return geometric_data_points


class GeometricDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, seed: int, max_atomic_num: int = 80, split: Tuple[int, int] = (0.9, 0.05),
                 train_path: str = None, separate_valid_path: str = None, separate_test_path: str = None, split_train: bool = False,
                 num_cores: Tuple[int, int, int] = (2, 0, 2), smiles_column_name: str = 'SMILES', label_column_name: str = 'Z-score',
                 use_standard_scaler=False):
        super().__init__()
        self.dataset = None
        self.train_path = train_path
        self.batch_size = batch_size
        self.seed = seed
        self.max_atomic_num = max_atomic_num
        self.split = split
        self.num_cores = num_cores
        self.separate_valid_path = separate_valid_path
        self.separate_test_path = separate_test_path
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.split_train = split_train
        self.use_standard_scaler = use_standard_scaler
        self.label_dims = None

        self.scaler = None
        if self.use_standard_scaler:
            train_df = pd.read_csv(self.train_path)
            train_data = train_df[self.label_column_name].values

            scaler = StandardScaler()
            if train_data.ndim == 1:
                scaler = scaler.fit(np.expand_dims(train_data, axis=1))
            else:
                scaler = scaler.fit(train_data)

            del train_data
            del train_df

            self.scaler = scaler

    def get_scaler(self):
        return self.scaler

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # Called on every GPU
        # Assumes prepare_data has been called

        self.val = None
        self.test = None
        if self.train_path:
            self.dataset = GraphMoleculeDataset(csv_path=self.train_path,
                                                max_atomic_num=self.max_atomic_num,
                                                smiles_column_name=self.smiles_column_name,
                                                label_column_name=self.label_column_name,
                                                scaler=self.scaler)
            self.label_dims = self.dataset.label_dims
            self.num_atom_features = self.dataset.num_atom_features
            self.num_bond_features = self.dataset.num_bond_features

        if self.separate_valid_path:
            self.val = GraphMoleculeDataset(csv_path=self.separate_valid_path,
                                            max_atomic_num=self.max_atomic_num,
                                            smiles_column_name=self.smiles_column_name,
                                            label_column_name=self.label_column_name,
                                            scaler=self.scaler)

        if self.separate_test_path:
            self.test = GraphMoleculeDataset(csv_path=self.separate_test_path,
                                             max_atomic_num=self.max_atomic_num,
                                             smiles_column_name=self.smiles_column_name,
                                             label_column_name=self.label_column_name,
                                             scaler=self.scaler)

        if (not self.separate_valid_path) and (not self.separate_test_path) and self.split_train:
            len_train, len_val = int(self.split[0] * len(self.dataset)), int(self.split[1] * len(self.dataset))
            len_test = len(self.dataset) - len_train - len_val
            assert len_train + len_val + len_test == len(self.dataset)

            self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [len_train, len_val, len_test],
                                                                            generator=torch.Generator().manual_seed(self.seed))
        elif self.train_path:
            self.train = self.dataset

    def train_dataloader(self, shuffle=True):
        if self.train:
            print('GeometricDataLoader: Train data is present.')
            return GeometricDataLoader(self.train, self.batch_size, shuffle=shuffle,
                                       num_workers=0 if not self.num_cores else self.num_cores[0])
        print('GeometricDataLoader: Train data is absent.')
        return None

    def val_dataloader(self):
        if self.val:
            print('GeometricDataLoader: Validation data is present.')
            return GeometricDataLoader(self.val, self.batch_size, shuffle=False,
                                       num_workers=0 if not self.num_cores else self.num_cores[1])
        print('GeometricDataLoader: Validation data is absent.')
        return None

    def test_dataloader(self):
        if self.test:
            print('GeometricDataLoader: Test data is present.')
            return GeometricDataLoader(self.test, self.batch_size, shuffle=False,
                                       num_workers=0 if not self.num_cores else self.num_cores[2])
        print('GeometricDataLoader: Test data is absent.')
        return None
