import re
import torch
import pytorch_lightning as pl
import numpy as np
import os
import torch_geometric.transforms as T


from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm.auto import tqdm
from pathlib import Path

from models.graph_models import GNN


MOLNET_DS = ['QM7', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipo', 'PCBA', 'MUV', 'HIV', 'BACE_CLS', 'BACE_REGR', 'BBBP', 'SIDER']
PYG_DS = ['ENZYMES', 'github_stargazers', 'PROTEINS_full', 'reddit_threads', 'REDDIT-BINARY', 'REDDIT-MULTI-12K', 'twitch_egos', 
                                        'TWITTER-Real-Graph-Partial', 'alchemy_full', 'COLORS-3', 'SYNTHETIC', 'SYNTHETICnew', 'Synthie', 'TRIANGLES', 'Cuneiform', 
                                        'COIL-DEL', 'COIL-RAG', 'AIDS', 'FRANKENSTEIN', 'IMDB-BINARY', 'MUTAG', 'Mutagenicity', 'YeastH', 'MalNetTiny']
PYG_OTHER_DS = ['ZINC', 'GNNBenchmark_MNIST', 'GNNBenchmark_CIFAR10']


def filter_model_only_args(all_argsdict, include_in_channels=True):
    model_keys = ['conv_type', 'gnn_intermediate_dim', 'gnn_output_node_dim', 'output_nn_intermediate_dim', 'readout', 
                  'learning_rate', 'gat_heads', 'gat_dropouts', 'pna_edge_dim', 'pna_num_towers', 'pna_num_pre_layers',
                  'pna_num_post_layers', 'dense_intermediate_dim', 'dense_output_graph_dim', 'set_transformer_k', 'use_vgae',
                  'set_transformer_dim_hidden', 'set_transformer_num_heads', 'set_transformer_layer_norm', 'set_transformer_num_inds', 
                  'janossy_num_perms', 'num_layers']
    if include_in_channels:
        model_keys.append('in_channels')

    return {k: all_argsdict[k] for k in all_argsdict.keys() if k in model_keys}


def deepchem_iterable_dataset_to_tensors(iter_dataset, use_cuda=False):
    aslist = list(iter_dataset)
    print('Processing DeepChem dataset for PyTorch...')

    new_dataset_as_list = []
    for batch in tqdm(aslist):
        graphs, ys, ws, smiles = batch
        ys = torch.from_numpy(ys.squeeze()).cuda() if use_cuda else torch.from_numpy(ys.squeeze())
        ws = torch.from_numpy(ws.squeeze()).cuda() if use_cuda else torch.from_numpy(ws.squeeze())
        for i in range(len(graphs)):
            graph_pyg = graphs[i].to_pyg_graph()
            gp = Data(x=graph_pyg.x.cuda() if use_cuda else graph_pyg.x, edge_index=graph_pyg.edge_index.cuda() if use_cuda else graph_pyg.edge_index, edge_attr=None, pos=None)
            gp.y = ys[i]
            gp.w = ws[i]
            gp.smiles = smiles[i]

            new_dataset_as_list.append(gp)

    return new_dataset_as_list


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    # Program-level args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--num_layers', type=int, default=2)

    # Optionally load from a saved checkpoint
    parser.add_argument('--ckpt_file', type=str)

    ### Data loading arguments ###
    # MoleculeNet dataset (required deepchem)
    parser.add_argument('--moleculenet_dataset', type=str)
    parser.add_argument('--moleculenet_random_split_seed', type=int)

    # PyTorch Geometric datasets
    parser.add_argument('--pyg_dataset', type=str)
    parser.add_argument('--pyg_dataset_splits_folder', type=str)
    parser.add_argument('--itr', type=int)

    # Custom molecular dataset
    parser.add_argument('--custom_dataset_train', type=str, required=False)
    parser.add_argument('--custom_dataset_validate', type=str, required=False)
    parser.add_argument('--custom_dataset_test', type=str, required=False)
    parser.add_argument('--custom_dataset_smiles_column', type=str, required=False)
    parser.add_argument('--custom_dataset_label_column', type=str, required=False)
    parser.add_argument('--custom_max_atomic_num', type=int, required=False)
    parser.add_argument('--custom_max_number_of_nodes', type=int, required=False)
    parser.add_argument('--custom_dataset_use_standard_scaler_on_label', dest='custom_dataset_use_standard_scaler_on_label', action='store_true',required=False)
    parser.add_argument('--custom_dataset_no_use_standard_scaler_on_label', dest='custom_dataset_use_standard_scaler_on_label', action='store_false',required=False)
    ### Data loading arguments ###
    
    # Required for both MoleculeNet and PyTorch Geometric datasets
    parser.add_argument('--dataset_download_dir', type=str)

    # Optionally, number of permutations seen during training for Janossy readouts
    parser.add_argument('--janossy_num_perms', type=int, default=25)

    parser = GNN.add_model_specific_args(parser)

    # Add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    argsdict = vars(args)

    # Check both variables are not set at the same time
    if argsdict['moleculenet_dataset'] is not None:
        assert argsdict['pyg_dataset'] is None and argsdict['custom_dataset_train'] is None, 'Can only have a single data source active (MoleculeNet OR PyTorch Geometric OR custom molecular dataset).'
    if argsdict['pyg_dataset'] is not None:
        assert argsdict['moleculenet_dataset'] is None and argsdict['custom_dataset_train'] is None, 'Can only have a single data source active (MoleculeNet OR PyTorch Geometric OR custom molecular dataset).'
    if argsdict['custom_dataset_train'] is not None:
        assert argsdict['moleculenet_dataset'] is None and argsdict['pyg_dataset'] is None, 'Can only have a single data source active (MoleculeNet OR PyTorch Geometric OR custom molecular dataset).'
        assert argsdict['custom_dataset_smiles_column'] is not None and argsdict['custom_dataset_label_column'] is not None and argsdict['custom_max_atomic_num'] is not None \
            and argsdict['custom_dataset_use_standard_scaler_on_label'] is not None, 'Must provide all necessary custom dataset settings.'

    if argsdict['moleculenet_dataset'] is not None:
        assert argsdict['moleculenet_dataset'] in MOLNET_DS, f'MoleculeNet dataset must be within the provided list: {str(MOLNET_DS)}'

    if argsdict['pyg_dataset'] is not None:
        assert argsdict['pyg_dataset'] in PYG_DS or argsdict['pyg_dataset'] in PYG_OTHER_DS,  f'PyG dataset must be within the provided list: {str(PYG_DS + PYG_OTHER_DS)}'

    if argsdict['moleculenet_dataset']:
        assert argsdict['moleculenet_random_split_seed'] is not None, 'Must provide MoleculeNet random seed for the splits.'

    if argsdict['pyg_dataset'] and argsdict['pyg_dataset'] not in ['GNNBenchmark_MNIST', 'GNNBenchmark_CIFAR10', 'ZINC']:
        assert argsdict['pyg_dataset_splits_folder'] is not None and argsdict['itr'] is not None, 'Must provide PyG random splits.'

    if argsdict['custom_dataset_train'] is None:
        assert argsdict['dataset_download_dir'] is not None, 'Must provide a download directory for the datasets.'
    assert argsdict['out_dir'] is not None, 'Must provide an output directory for the checkpoints and saved data.'

    assert argsdict['num_layers'] is not None and argsdict['num_layers'] > 1, 'Must provide a number of layers that is > 1.'

    if argsdict['readout'] in ['janossy dense', 'janossy gru']:
        assert argsdict['janossy_num_perms'] is not None, 'Must provide the --janossy_num_perms argument when using Janossy readouts.'

    if argsdict['conv_type'] in ['GAT', 'GATv2']:
        assert argsdict['gat_heads'] is not None and argsdict['gat_dropouts'] is not None, 'Must provide the --gat_heads and --gat_dropouts arguments for GAT and GATv2.'

    if argsdict['conv_type'] == 'PNA':
        assert argsdict['pna_num_towers'] is not None and argsdict['pna_num_pre_layers'] is not None and argsdict['pna_num_post_layers'] is not None, 'Must provide the --pna_num_towers --pna_num_pre_layers and --pna_num_post_layers arguments for PNA.'
        assert argsdict['gnn_output_node_dim'] % argsdict['pna_num_towers'] == 0, '--gnn_output_node_dim must be divisible by --pna_num_towers.'
        assert argsdict['gnn_intermediate_dim'] % argsdict['pna_num_towers'] == 0, '--gnn_intermediate_dim must be divisible by --pna_num_towers.'

    if argsdict['pyg_dataset'] == 'MalNetTiny':
        import warnings
        warnings.warn('MalNetTiny loading will crash if using less than 64GB of RAM. Also, CUDA models can crash if the batch size is not reduced from 32.')

    if argsdict['pyg_dataset'] in ['REDDIT-BINARY', 'REDDIT-MULTI-12K']:
        import warnings
        warnings.warn('CUDA models can crash if the batch size is not reduced from 32.')

    # ------------
    # data
    # ------------

    ### WARINING: Some splitters (e.g. ScaffoldSplitter()) do not change according to seed

    if argsdict['moleculenet_dataset'] is not None:
        import deepchem as dc
        in_channels = 30

        if argsdict['moleculenet_dataset'] == 'QM7':
            tasks, datasets, transformers = dc.molnet.load_qm7(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomStratifiedSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MAE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'QM8':
            tasks, datasets, transformers = dc.molnet.load_qm8(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MAE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'QM9':
            tasks, datasets, transformers = dc.molnet.load_qm9(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MAE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'ESOL':
            tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                   splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MSE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'FreeSolv':
            tasks, datasets, transformers = dc.molnet.load_sampl(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                 splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MSE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'Lipo':
            tasks, datasets, transformers = dc.molnet.load_lipo(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MSE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'PCBA':
            tasks, datasets, transformers = dc.molnet.load_pcba(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'


        elif argsdict['moleculenet_dataset'] == 'MUV':
            tasks, datasets, transformers = dc.molnet.load_muv(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'


        elif argsdict['moleculenet_dataset'] == 'HIV':
            tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.ScaffoldSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'


        elif argsdict['moleculenet_dataset'] == 'BACE_CLS':
            tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.ScaffoldSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'


        elif argsdict['moleculenet_dataset'] == 'BACE_REGR':
            tasks, datasets, transformers = dc.molnet.load_bace_regression(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                           splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.ScaffoldSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MSE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'BBBP':
            tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.ScaffoldSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'


        elif argsdict['moleculenet_dataset'] == 'SIDER':
            tasks, datasets, transformers = dc.molnet.load_sider(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                 splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'


        train_dataset, validation_dataset, test_dataset = datasets
        train_dataset = train_dataset.make_pytorch_dataset(epochs=1, deterministic=True, batch_size=len(train_dataset))
        validation_dataset = validation_dataset.make_pytorch_dataset(epochs=1, deterministic=True, batch_size=len(validation_dataset))
        test_dataset = test_dataset.make_pytorch_dataset(epochs=1, deterministic=True, batch_size=len(test_dataset))

        train_dataset = deepchem_iterable_dataset_to_tensors(train_dataset, use_cuda=argsdict['gpus'] == 1)
        validation_dataset = deepchem_iterable_dataset_to_tensors(validation_dataset, use_cuda=argsdict['gpus'] == 1)
        test_dataset = deepchem_iterable_dataset_to_tensors(test_dataset, use_cuda=argsdict['gpus'] == 1)

        num_train_workers = (0, 0, 0)
        train_loader = DataLoader(train_dataset, batch_size=argsdict['batch_size'], num_workers=num_train_workers[0])
        validation_loader = DataLoader(validation_dataset, batch_size=argsdict['batch_size'], num_workers=num_train_workers[1])
        test_loader = DataLoader(test_dataset, batch_size=argsdict['batch_size'], num_workers=num_train_workers[2])

        num_tasks = len(tasks)

        if argsdict['conv_type'] == 'PNA':
            print('Computing max degree for PNA...')
            degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in train_dataset])
            degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in train_dataset])
            deg = int(max(degree_0, degree_1)) + 1
        else:
            deg = 0

    elif argsdict['pyg_dataset'] is not None and argsdict['pyg_dataset'] in ['GNNBenchmark_MNIST', 'GNNBenchmark_CIFAR10']:
        import torch_geometric.datasets as ds
        root = argsdict['dataset_download_dir']

        if argsdict['pyg_dataset'] in ['GNNBenchmark_MNIST', 'GNNBenchmark_CIFAR10']:
            if argsdict['pyg_dataset'] == 'GNNBenchmark_MNIST':
                name = 'MNIST'
            elif argsdict['pyg_dataset'] == 'GNNBenchmark_CIFAR10':
                name = 'CIFAR10'

            train_dataset = ds.GNNBenchmarkDataset(root=root, name=name, split='train')
            validation_dataset = ds.GNNBenchmarkDataset(root=root, name=name, split='val')
            test_dataset = ds.GNNBenchmarkDataset(root=root, name=name, split='test')

            loss_metric = 'CrossEntropyLoss'
            task_type = 'multi_classification'

            in_channels = train_dataset.num_node_features
            num_tasks = train_dataset.num_classes

        if argsdict['conv_type'] == 'PNA':
            print('Computing max degree for PNA...')
            degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in train_dataset])
            degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in train_dataset])
            deg = int(max(degree_0, degree_1)) + 1
        else:
            deg = 0

        train_loader = DataLoader(train_dataset, batch_size=argsdict['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=argsdict['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=argsdict['batch_size'])

        print(f'Using {argsdict["pyg_dataset"]} dataset from PyTorch Geometric.')

    elif argsdict['pyg_dataset'] is not None and argsdict['pyg_dataset'] == 'ZINC':
        import torch_geometric.datasets as ds
        root = argsdict['dataset_download_dir']

        train_dataset = ds.ZINC(root=root, split='train', subset=False)
        validation_dataset = ds.ZINC(root=root, split='val', subset=False)
        test_dataset = ds.ZINC(root=root, split='test', subset=False)

        loss_metric = 'MSE'
        task_type = 'regression'

        in_channels = train_dataset.num_node_features
        num_tasks = 1

        if argsdict['conv_type'] == 'PNA':
            print('Computing max degree for PNA...')
            degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in train_dataset])
            degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in train_dataset])
            deg = int(max(degree_0, degree_1)) + 1
        else:
            deg = 0

    elif argsdict['pyg_dataset'] is not None and  argsdict['pyg_dataset'] in ['ENZYMES', 'github_stargazers', 'PROTEINS_full', 'reddit_threads',\
                                                        'REDDIT-BINARY', 'REDDIT-MULTI-12K', 'twitch_egos', 'TWITTER-Real-Graph-Partial', 'alchemy_full',\
                                                        'COLORS-3', 'SYNTHETIC', 'SYNTHETICnew', 'Synthie', 'TRIANGLES', 'Cuneiform', 'COIL-DEL', 'COIL-RAG',\
                                                        'AIDS', 'FRANKENSTEIN', 'IMDB-BINARY', 'MUTAG', 'Mutagenicity', 'YeastH', 'MalNetTiny']:
        import torch_geometric.datasets as ds
        root = argsdict['dataset_download_dir']

        dir_path = os.path.join(argsdict['pyg_dataset_splits_folder'], argsdict['pyg_dataset'])
        perm = np.load(os.path.join(dir_path, f'random_permutation_{argsdict["itr"]}.npy'))

        if argsdict['pyg_dataset'] != 'MalNetTiny':
            dataset = ds.TUDataset(root=root, name=argsdict['pyg_dataset'], use_node_attr=True)
        else:
            dataset = ds.MalNetTiny(root=root)

        degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in dataset])
        degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in dataset])
        deg = int(max(degree_0, degree_1)) + 1

        if argsdict['pyg_dataset'] in ['github_stargazers', 'IMDB-BINARY', 'reddit_threads', 'REDDIT-BINARY', 'REDDIT-MULTI-12K',\
                                        'twitch_egos', 'TRIANGLES', 'TWITTER-Real-Graph-Partial', 'MalNetTiny']:
            if argsdict['pyg_dataset'] != 'MalNetTiny':
                dataset = ds.TUDataset(root=root, name=argsdict['pyg_dataset'], transform=T.OneHotDegree(max_degree=int(max(degree_0, degree_1))))
            else:
                dataset = ds.MalNetTiny(root=root, transform=T.OneHotDegree(max_degree=int(max(degree_0, degree_1))))

            in_channels = int(max(degree_0, degree_1)) + 1
        else:
            in_channels = dataset.num_node_features

        dataset = dataset.index_select(perm)

        # WARNING: This step can crash for MalNetTiny if the amount of RAM is < 64GB
        dataset_as_numpy = np.asarray(dataset, dtype=object)
        train, validate, test = np.split(dataset_as_numpy, [int(0.8 * len(dataset_as_numpy)), int(0.9 * len(dataset_as_numpy))])

        train_dataset = [Data(**{item[0]: item[1] for item in data}) for data in train]
        validation_dataset = [Data(**{item[0]: item[1] for item in data}) for data in validate]
        test_dataset = [Data(**{item[0]: item[1] for item in data}) for data in test]

        if argsdict['pyg_dataset'] in ['github_stargazers', 'IMDB-BINARY', 'PROTEINS_full', 'reddit_threads', 'REDDIT-BINARY', 'REDDIT-MULTI-12K'\
                                        'twitch_egos', 'TWITTER-Real-Graph-Partial', 'SYNTHETIC', 'SYNTHETICnew', 'AIDS', 'FRANKENSTEIN', 'Mutagenicity',\
                                        'MUTAG', 'YeastH']:
            task_type = 'binary_classification'
            loss_metric = 'BCEWithLogits'

        elif argsdict['pyg_dataset'] == 'alchemy_full':
            task_type = 'regression'
            loss_metric = 'MSE'

        else:
            task_type = 'multi_classification'
            loss_metric = 'CrossEntropyLoss'

        num_tasks = dataset.num_classes
        if num_tasks == 2:
            num_tasks = 1

    elif argsdict['custom_dataset_train'] is not None:
        from utils.data_loading import GeometricDataModule
        custom_dataset = GeometricDataModule(batch_size=argsdict['batch_size'], seed=0,
                                            train_path=argsdict['custom_dataset_train'],
                                            separate_valid_path=argsdict['custom_dataset_validate'],
                                            separate_test_path=argsdict['custom_dataset_test'],
                                            split_train=False, num_cores=(0, 0, 0),
                                            smiles_column_name=argsdict['custom_dataset_smiles_column'],
                                            label_column_name=argsdict['custom_dataset_label_column'],
                                            use_standard_scaler=argsdict['custom_dataset_use_standard_scaler_on_label'],
                                            max_atomic_num=argsdict['custom_max_atomic_num'])

        custom_dataset.prepare_data()
        custom_dataset.setup()
        train_loader = custom_dataset.train_dataloader()
        validation_loader = custom_dataset.val_dataloader()
        test_loader = custom_dataset.test_dataloader()
        num_tasks = custom_dataset.label_dims
        in_channels = argsdict['custom_max_atomic_num'] + 27

        loss_metric = 'MSE'
        task_type = 'regression'
        num_tasks = 1
        train_dataset = custom_dataset.dataset


        if argsdict['conv_type'] == 'PNA':
            print('Computing max degree for PNA...')
            degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in train_dataset])
            degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in train_dataset])
            deg = int(max(degree_0, degree_1)) + 1
        else:
            deg = 0


    if argsdict['pyg_dataset']:
        train_loader = DataLoader(train_dataset, batch_size=argsdict['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=argsdict['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=argsdict['batch_size'])

    num_nodes_train = max([d.x.shape[0] for d in train_dataset])

    if argsdict['custom_dataset_validate']:
        num_nodes_val = max([d.x.shape[0] for d in custom_dataset.val])
    else:
        num_nodes_val = 0

    if argsdict['custom_dataset_test']:
        num_nodes_test = max([d.x.shape[0] for d in custom_dataset.test])
    else:
        num_nodes_test = 0

    print(argsdict['custom_max_number_of_nodes'])
    if argsdict['custom_max_number_of_nodes'] is None:
        print('Computing the maximum number of nodes...')
        num_nodes = np.max((num_nodes_train, num_nodes_val, num_nodes_test))
    else:
        num_nodes = argsdict['custom_max_number_of_nodes']

    print(f'Size of training dataset = {len(train_loader)}.')
    if validation_loader is not None:
        print(f'Size of validation dataset = {len(validation_loader)}.')
    if test_loader is not None:
        print(f'Size of test dataset = {len(test_loader)}.')


    # ------------
    # model
    # ------------
    print('Creating model...')
    model = GNN(in_channels=in_channels, max_num_nodes_in_graph=num_nodes,
        **filter_model_only_args(argsdict, include_in_channels=False), output_nn_out_dim=num_tasks, loss_metric=loss_metric, 
        task_type=task_type, train_dataset=train_dataset, dataset_degree=deg, use_cuda=argsdict['gpus'] == 1)

    print('Model summary: ')
    print(model)

    # ------------
    # training
    # ------------

    monitor = 'validation_total_loss' if argsdict['custom_dataset_train'] is None else 'train_total_loss'

    checkpoint = ModelCheckpoint(
            monitor=monitor,
            dirpath=argsdict['out_dir'],
            filename='gnn-{epoch:03d}-{validation_total_loss:.5f}' if argsdict['custom_dataset_train'] is None else 'gnn-{epoch:03d}-{train_total_loss:.5f}',
            save_top_k=1 if argsdict['custom_dataset_train'] is None else -1,
            mode='min',
        )

    if argsdict['custom_dataset_train'] is None:
        early_stopping = EarlyStopping(
                monitor=monitor,
                min_delta=0.00,
                patience=30,
                verbose=False,
                mode='min'
            )

    callbacks = [checkpoint, early_stopping] if argsdict['custom_dataset_train'] is None else [checkpoint]

    print('Creating Trainer...')
    logs_path = os.path.join(argsdict['out_dir'], 'logs/')
    Path(logs_path).mkdir(exist_ok=True, parents=True)
    logger = CSVLogger(save_dir=logs_path, name='gnn_logs')

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    print('Starting training...')

    loaders = (train_loader, validation_loader) if argsdict['custom_dataset_train'] is None else (train_loader,)

    if not argsdict['ckpt_file']:
        trainer.fit(model, *loaders)
    else:
        trainer.fit(model, *loaders, ckpt_path=argsdict['ckpt_file'])


    # ------------
    # testing
    # ------------
    should_test_flag = (argsdict['moleculenet_dataset'] is not None) or (argsdict['pyg_dataset'] is not None) or (argsdict['custom_dataset_train'] is not None and argsdict['custom_dataset_test'] is not None)
    if should_test_flag:
        trainer.test(model, dataloaders=test_loader)

    # ------------
    # saving
    # ------------
    if should_test_flag:
        save_out_path = os.path.join(argsdict['out_dir'], 'saved_data')
        Path(save_out_path).mkdir(exist_ok=True, parents=True)


        np.save(os.path.join(save_out_path, 'train_metrics_per_epoch.npy'), model.train_metrics_per_epoch)
        np.save(os.path.join(save_out_path, 'validation_metrics_per_epoch.npy'), model.validation_metrics_per_epoch)
        np.save(os.path.join(save_out_path, 'test_metrics_per_epoch.npy'), model.test_metrics_per_epoch)

        # np.save(os.path.join(save_out_path, 'train_graphs_per_epoch.npy'), model.train_graphs_per_epoch)
        # np.save(os.path.join(save_out_path, 'validation_graphs_per_epoch.npy'), model.validation_graphs_per_epoch)
        np.save(os.path.join(save_out_path, 'test_graphs_per_epoch.npy'), model.test_graphs_per_epoch)

        # np.save(os.path.join(save_out_path, 'train_predictions_per_epoch.npy'), model.train_outputs)
        # np.save(os.path.join(save_out_path, 'validation_predictions_per_epoch.npy'), model.validation_outputs)
        np.save(os.path.join(save_out_path, 'test_predictions_per_epoch.npy'), model.test_outputs)


if __name__ == '__main__':
    main()
