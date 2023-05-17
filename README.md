# Graph Neural Networks with Adaptive Readouts

The code and included data can be used to reproduce the experiments described in the [*Graph Neural Networks with Adaptive Readouts*](https://papers.nips.cc/paper_files/paper/2022/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html) paper (NeurIPS 2022).

## 🎉 Native PyTorch Geometric support
Adaptive readouts are now available directly in PyTorch Geometric 2.3.0 as [aggregation operators](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators)! These are drop-in replacements for simpler alternatives such as sum, mean, or maximum in the updated PyG aggregation workflow. The operators described in the paper are [MLPAggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.MLPAggregation.html#torch_geometric.nn.aggr.MLPAggregation), [SetTransformerAggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.SetTransformerAggregation.html), [GRUAggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.GRUAggregation.html#torch_geometric.nn.aggr.GRUAggregation), and [DeepSetsAggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.DeepSetsAggregation.html).

## Abstract
An effective aggregation of node features into a graph-level representation via readout functions is an essential step in numerous learning tasks involving graph neural networks. Typically, readouts are simple and non-adaptive functions designed such that the resulting hypothesis space is permutation invariant. Prior work on deep sets indicates that such readouts might require complex node embeddings that can be difficult to learn via standard neighborhood aggregation schemes. Motivated by this, we investigate the potential of adaptive readouts given by neural networks that do not necessarily give rise to permutation invariant hypothesis spaces. We argue that in some problems such as binding affinity prediction where molecules are typically presented in a canonical form it might be possible to relax the constraints on permutation invariance of the hypothesis space and learn a more effective model of the affinity by employing an adaptive readout function. Our empirical results demonstrate the effectiveness of neural readouts on more than 40 datasets spanning different domains and graph characteristics. Moreover, we observe a consistent improvement over standard readouts (i.e., sum, max, and mean) relative to the number of neighborhood aggregation iterations and different convolutional operators.

## Cite
If you find the idea and/or implementation of adaptive readouts useful in your work, a citation is appreciated.

```
@inproceedings{NEURIPS2022_7caf9d25,
    author = {Buterez, David and Janet, Jon Paul and Kiddle, Steven J and Oglic, Dino and Li\`{o}, Pietro},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
    pages = {19746--19758},
    publisher = {Curran Associates, Inc.},
    title = {Graph Neural Networks with Adaptive Readouts},
    url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/7caf9d251b546bc78078b35b4a6f3b7e-Paper-Conference.pdf},
    volume = {35},
    year = {2022}
}
```

We have leveraged these techniques to accelerate and improve drug discovery workflows:

```
 @article{
    buterez_janet_kiddle_liò_2022,
    place={Cambridge},
    title={Multi-fidelity machine learning models for improved high-throughput screening predictions},
    DOI={10.26434/chemrxiv-2022-dsbm5-v2},
    journal={ChemRxiv},
    publisher={Cambridge Open Engage},
    author={Buterez, David and Janet, Jon Paul and Kiddle, Steven and Liò, Pietro},
    year={2022}
 }
```

```
@article{doi:10.1021/acs.jcim.2c01569,
    author = {Buterez, David and Janet, Jon Paul and Kiddle, Steven J. and Liò, Pietro},
    title = {MF-PCBA: Multifidelity High-Throughput Screening Benchmarks for Drug Discovery and Machine Learning},
    journal = {Journal of Chemical Information and Modeling},
    volume = {0},
    number = {0},
    pages = {null},
    year = {0},
    doi = {10.1021/acs.jcim.2c01569},

    URL = {
        https://doi.org/10.1021/acs.jcim.2c01569
    },
    eprint = {
        https://doi.org/10.1021/acs.jcim.2c01569 
    }
}
```

## Implementation
The code specific to the neural/adaptive readouts implementation is available in [code/models/graph_models.py](https://github.com/davidbuterez/gnn-neural-readouts/blob/main/code/models/graph_models.py#L321-L356).

## Supplementary materials

The `Supplementary materials` directory contains the files referenced in the paper:
1. Random seeds and splits for all the datasets (see below).
2. Metrics for all classification and regression datasets (separately), for each GNN layer type, readout, and random split.

## 3D UMAP visualization

All the saved embeddings are not included in the archive due to taking too much space. However, the saved embeddings for AID1949 (the smallest of the three datasets) are provided for the Sum, Dense, and Set Transformer readouts in the `UMAP_graph_embeddings` directory. The code to visualize the embeddings (as shown in the paper) is available in the notebook `visualization.ipynb`.

## QM9 random permutations

The permutations for the 50 molecules are provided in the `QM9_permutations` directory. Inside, the `QM9_arbitrary_random_permutations` directory corresponds to the arbitrary permutations strategy presented in the paper, and the `QM9_permutations_from_random_SMILES` directory corresponds to the strategy involving random non-canonical SMILES. Within the two directories there are 50 directories corresponding to the 50 molecules, each named after their position (index) in QM9. Each molecule directory contains an `original_data.npy` file that has the molecule index, the original (non-permuted) node feature matrix, and the original (non-permuted) edge index (adjacency). Each molecule directory also contains `<int>.npy` files (count starting from 0), each containing a node matrix permutation and the corresponding permuted edge index.

## Code

The `code` directory contains:

1. Train and test code for the DeepChem and PyTorch Geometric datasets mentioned in the paper (random seeds provided).
2. Train and test code for custom molecular datasets (both GNN/VGAE). This corresponds to the bioaffinity (high-throughput screening) experiments, presented in the paper with the train loss plots and resulting UMAP embeddings.
3. Instructions and examples of using the provided code.
4. Code to visualize the 3D UMAP graph embeddings (same as shown in the paper) — all the embeddings themselves could not be included in this submission due to the large size, but a few saved embeddings are provided (see corresponding section below).

## Installation
The code requires PyTorch, PyTorch Geometric, PyTorch Lightning, as well as DeepChem and RDKit for many datasets. An example `conda` environment is provided in `torch_geometric_env.yaml`.

Instructions to install the latest versions from scratch (assuming a CUDA-enabled machine):

1. Install PyTorch:

	`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
2. Install PyTorch Geometric:

	`conda install pyg -c pyg`

3. Install PyTorch Lightning:

	`pip install pytorch-lightning`
4. (Optional but recommended) Install DeepChem with the (slightly modified) recommended commands:

    `pip install --pre deepchem`

	`pip install tensorflow~=2.6`

5. (Optional but recommended) If the previous step does not correctly install RDKit, it can be quickly installed with:

	`pip install rdkit-pypi`


## Experiments
### MoleculeNet datasets
The MoleculeNet models are trained with random seeds that are provided in the file `molnet_random_seeds.npy` from 'Supplementary materials' and listed in `view_random_seeds.ipynb`.

The examples below are for QM9, one of the random seeds, and GCN layers. Set `--gpus 1` to 0 to use the CPU instead of the GPU (CUDA).

The simplest command (for the sum/mean/max readouts) is the following (the `out_dir` will be created if it does not exist; `download_dir` will be used by DeepChem to download and featurize the dataset):

##### Sum/Mean/Max (set `--readout` to `sum`, `mean`, or `max`)

```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir
```

##### Dense (`--readout dense`, add `--dense_intermediate_dim <int>` and `--dense_output_graph_dim <int>`)

```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout dense --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --dense_intermediate_dim 256 --dense_output_graph_dim 128
```

##### GRU (`--readout gru`)
```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout gru --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir
```

##### Set Transformer (`--readout "set transformer"`, add `--set_transformer_k <int>`, `--set_transformer_dim_hidden <int>`, `--set_transformer_num_heads <int>`, and `--no-set_transformer_layer_norm` or `--set_transformer_layer_norm`)

```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout "set transformer" --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --set_transformer_k 8 --set_transformer_dim_hidden 512 --set_transformer_num_heads 8 --no-set_transformer_layer_norm
```

##### Janossy Dense (`--readout "janossy dense"`, add `--janossy_num_perms <int>`)

```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout "janossy dense" --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --dense_intermediate_dim 256 --dense_output_graph_dim 128 --janossy_num_perms 25
```

##### Janossy GRU (`--readout "janossy gru"`, add `--janossy_num_perms <int>`)

```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout "janossy gru" --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --janossy_num_perms 25
```

</br>

### Different layer types
##### GAT/GATv2  (requires `--gat_heads <int>` and `--gat_dropouts <float>`)
```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GAT --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --gat_heads 5 --gat_dropouts 0.2
```
</br>


##### PNA (requires `--pna_num_towers <int>`, `--pna_num_pre_layers <int>`, and `--pna_num_post_layers <int>`)
##### Also, `--gnn_output_node_dim` and `--gnn_intermediate_dim` must be divisible by `--pna_num_towers`

```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type PNA --gnn_intermediate_dim 65 --gnn_output_node_dim 35 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pna_num_towers 5 --pna_num_pre_layers 1 --pna_num_post_layers 1
```
</br>

### Different numbers of layers (`--num_layers <int>`)
```
python run.py  --moleculenet_dataset QM9 --moleculenet_random_split_seed 238442 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --num_layers 4
```
</br>

### Datasets without random splits:
`--dataset_download_dir` is used by PyTorch Geometric to download the datasets.

##### ZINC
```
python run.py  --pyg_dataset ZINC --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir
```

##### GNNBenchmark_MNIST
```
python run.py  --pyg_dataset GNNBenchmark_MNIST --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir
```

##### GNNBenchmark_CIFAR10
```
python run.py  --pyg_dataset GNNBenchmark_CIFAR10 --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir
```

</br>

###  PyTorch Geometric datasets with random splits (requires `--pyg_dataset_splits_folder` and `--itr`):
`--pyg_dataset_splits_folder` needs to point to the directory containing the random splits (included in the archive). `--itr` indicates which split to use.

Examples for different datasets:

```
python run.py  --pyg_dataset ENZYMES --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pyg_dataset_splits_folder /home/david/Projects/Molnet_nn_gru/pyg_shuffled_datasets/ --itr 0
```

```
python run.py  --pyg_dataset MUTAG --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pyg_dataset_splits_folder /home/david/Projects/Molnet_nn_gru/pyg_shuffled_datasets/ --itr 0
```

```
python run.py  --pyg_dataset REDDIT-BINARY --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pyg_dataset_splits_folder /home/david/Projects/Molnet_nn_gru/pyg_shuffled_datasets/ --itr 0
```

```
python run.py  --pyg_dataset alchemy_full --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pyg_dataset_splits_folder /home/david/Projects/Molnet_nn_gru/pyg_shuffled_datasets/ --itr 0
```

```
python run.py  --pyg_dataset AIDS --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pyg_dataset_splits_folder /home/david/Projects/Molnet_nn_gru/pyg_shuffled_datasets/ --itr 0
```

```
python run.py  --pyg_dataset MalNetTiny --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --dataset_download_dir download_dir --pyg_dataset_splits_folder /home/david/Projects/Molnet_nn_gru/pyg_shuffled_datasets/ --itr 0
```

</br>

### Custom molecular datasets (only train; entire dataset used)
#### AID1949

##### VGAE
```
python run.py  --custom_dataset_train /home/david/NN_AGG_code_to_submit/code_to_submit/submission/pubchem_datasets/AID1949/SD.csv --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --custom_dataset_smiles_column neut-smiles --custom_dataset_label_column SD --custom_max_atomic_num 53 --custom_dataset_use_standard_scaler_on_label --custom_max_number_of_nodes 44 --use_vgae
```

##### No VGAE
```
python run.py  --custom_dataset_train /home/david/NN_AGG_code_to_submit/code_to_submit/submission/pubchem_datasets/AID1949/SD.csv --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --custom_dataset_smiles_column neut-smiles --custom_dataset_label_column SD --custom_max_atomic_num 53 --custom_dataset_use_standard_scaler_on_label --custom_max_number_of_nodes 44
```
</br>

#### AID449762
##### VGAE
```
python run.py  --custom_dataset_train /home/david/NN_AGG_code_to_submit/code_to_submit/submission/pubchem_datasets/AID449762/SD.csv --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --custom_dataset_smiles_column neut-smiles --custom_dataset_label_column SD --custom_max_atomic_num 80 --custom_dataset_use_standard_scaler_on_label --custom_max_number_of_nodes 101 --use_vgae
```
</br>

#### AID602261
##### VGAE
```
python run.py  --custom_dataset_train /home/david/NN_AGG_code_to_submit/code_to_submit/submission/pubchem_datasets/AID602261/SD.csv --out_dir out_dir  --batch_size 32 --conv_type GCN --gnn_intermediate_dim 64 --gnn_output_node_dim 32 --output_nn_intermediate_dim 32 --readout sum --learning_rate 0.0001 --max_epochs 100000 --min_epochs 10 --gpus 1 --custom_dataset_smiles_column neut-smiles --custom_dataset_label_column SD --custom_max_atomic_num 80 --custom_dataset_use_standard_scaler_on_label --custom_max_number_of_nodes 101 --use_vgae
```
