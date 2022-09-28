import torch
import numpy as np
import scipy as sp
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric

from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, GRU
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, PNAConv, global_add_pool, global_mean_pool, global_max_pool, InnerProductDecoder
from torch_geometric.utils import degree, to_dense_batch, negative_sampling, remove_self_loops, add_self_loops
from tqdm.auto import tqdm


EPS = 1e-15
MAX_LOGSTD = 10


def get_regression_metrics(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    maxer = np.max(np.abs(errors))
    r2, pval = np.power(sp.stats.pearsonr(y_true.flatten(), y_pred.flatten()), 2)

    return (mae, rmse, maxer, r2, np.sqrt(pval))


def get_classification_metrics(y_true, y_pred, digits=6):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    # ROC AUC not defined if a single label is present
    except ValueError:
        roc_auc = None

    return confusion_matrix(y_true, y_pred), roc_auc, classification_report(y_true, y_pred, digits=digits), matthews_corrcoef(y_true, y_pred)


def get_degrees(train_dataset_as_list, dataset_degree, use_cuda=True):
    deg = torch.zeros(dataset_degree, dtype=torch.long, device=torch.device('cuda') if use_cuda else torch.device('cpu'))

    print('Computing degrees for PNA...')
    for data in tqdm(train_dataset_as_list):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        bincount = torch.bincount(d, minlength=deg.numel())
        deg += bincount.to(torch.device('cuda')) if use_cuda else bincount

    return deg


def _vgae_recon_loss(z, pos_edge_index, neg_edge_index=None):
    decoder = InnerProductDecoder()

    # PyG implementation
    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS
    ).mean()

    # Do not include self-loops in negative samples
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 -
                          decoder(z, neg_edge_index, sigmoid=True) +
                          EPS).mean()

    return pos_loss + neg_loss


def _vgae_kl_loss(mu, logstd):
    # PyG implementation
    logstd = logstd.clamp(max=MAX_LOGSTD)
    return -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
    )


class GNN(pl.LightningModule):
    def __init__(self,
                 conv_type: str,
                 in_channels: int,
                 gnn_intermediate_dim: int,
                 gnn_output_node_dim: int,
                 output_nn_intermediate_dim: int,
                 output_nn_out_dim: int,
                 task_type: str,
                 readout: str,
                 loss_metric: str,
                 learning_rate: float,
                 num_layers: int,
                 use_vgae: str,
                 max_num_nodes_in_graph: int = None,
                 gat_heads: int = None,
                 gat_dropouts: int = None,
                 train_dataset: torch.utils.data.Dataset = None,
                 dataset_degree: int = None,
                 pna_edge_dim: int = None,
                 pna_num_towers: int = 5,
                 pna_num_pre_layers: int = 1,
                 pna_num_post_layers: int = 1,
                 dense_intermediate_dim: int = None,
                 dense_output_graph_dim: int = None,
                 set_transformer_k: int = None,
                 set_transformer_dim_hidden: int = None,
                 set_transformer_num_heads: int = None,
                 set_transformer_layer_norm: bool = None,
                 set_transformer_num_inds: int = None,
                 janossy_num_perms: int = None,
                 use_cuda: bool = False,
                 ):
        super(GNN, self).__init__()

        # self.save_hyperparameters()

        self.conv_type = conv_type
        self.in_channels = in_channels
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.gnn_output_node_dim = gnn_output_node_dim
        self.output_nn_intermediate_dim = output_nn_intermediate_dim
        self.output_nn_out_dim = output_nn_out_dim
        self.task_type = task_type
        self.readout = readout
        self.loss_metric = loss_metric
        self.learning_rate = learning_rate

        self.max_num_nodes_in_graph = max_num_nodes_in_graph
        self.gat_heads = gat_heads
        self.gat_dropouts = gat_dropouts
        self.dataset_degree = dataset_degree # Need this for PNA
        self.train_dataset = train_dataset # Need this for PNA
        self.pna_edge_dim = pna_edge_dim
        self.pna_num_towers = pna_num_towers
        self.pna_num_pre_layers = pna_num_pre_layers
        self.pna_num_post_layers = pna_num_post_layers

        self.dense_intermediate_dim = dense_intermediate_dim
        self.dense_output_graph_dim = dense_output_graph_dim

        self.set_transformer_k = set_transformer_k
        self.set_transformer_dim_hidden = set_transformer_dim_hidden
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_layer_norm = set_transformer_layer_norm
        self.set_transformer_num_inds = set_transformer_num_inds

        self.janossy_num_perms = janossy_num_perms
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.use_vgae = use_vgae

        # Graph embedding dimension is the same as the node dimension if using sum/mean/max
        # If using a dense readout it can be anything
        # When using GRU this also needs to be set
        self.graph_dim = self.gnn_output_node_dim if not self.dense_output_graph_dim else self.dense_output_graph_dim

        # Storage
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}

        self.train_graphs_per_epoch = {}
        self.validation_graphs_per_epoch = {}
        self.test_graphs_per_epoch = {}

        # Input assertions
        assert self.conv_type in ['GCN', 'GAT', 'GATv2', 'GIN', 'PNA']
        assert self.task_type in ['regression', 'binary_classification', 'multi_classification']
        assert self.readout in ['sum', 'mean', 'max', 'dense', 'set transformer', 'gru', 'janossy dense', 'janossy gru']
        assert self.loss_metric in ['MAE', 'MSE', 'BCEWithLogits', 'CrossEntropyLoss']

        print(f'Training with {self.num_layers} layers.')
        if self.use_vgae:
            print(f'Using VGAE architecture with two additional mu and sigma layers.')

        # Convolutional layers
        convs = []

        # GCN
        if self.conv_type == 'GCN':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GCNConv(in_channels=self.in_channels, out_channels=self.gnn_intermediate_dim, cached=False,  normalize=True), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GCNConv(in_channels=self.gnn_intermediate_dim, out_channels=self.gnn_intermediate_dim, cached=False, normalize=True), 'x, edge_index -> x'))
                else:
                    convs.append((GCNConv(in_channels=self.gnn_intermediate_dim, out_channels=self.gnn_output_node_dim, cached=False, normalize=True), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

            if self.use_vgae:
                self.conv_mu = GCNConv(in_channels=self.gnn_output_node_dim, out_channels=self.gnn_output_node_dim, cached=False, normalize=True)
                self.conv_logstd = GCNConv(in_channels=self.gnn_output_node_dim, out_channels=self.gnn_output_node_dim, cached=False, normalize=True)

        # GAT
        if self.conv_type == 'GAT':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GATConv(in_channels=self.in_channels, out_channels=self.gnn_intermediate_dim, heads=self.gat_heads,
                                            concat=True, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GATConv(in_channels=self.gnn_intermediate_dim * self.gat_heads, out_channels=self.gnn_intermediate_dim,
                                            heads=self.gat_heads, concat=True, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                else:
                    convs.append((GATConv(in_channels=self.gnn_intermediate_dim * self.gat_heads, out_channels=self.gnn_output_node_dim,
                                            heads=self.gat_heads, concat=False, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

            if self.use_vgae:
                self.conv_mu = GATConv(in_channels=self.gnn_output_node_dim * self.gat_heads[1], out_channels=self.gnn_output_node_dim,
                                        heads=self.gat_heads[2], concat=False, dropout=self.gat_dropouts[2])
                self.conv_logstd = GATConv(in_channels=self.gnn_output_node_dim * self.gat_heads[1], out_channels=self.gnn_output_node_dim,
                                            heads=self.gat_heads[3], concat=False, dropout=self.gat_dropouts[3])

        # GATv2
        if self.conv_type == 'GATv2':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GATv2Conv(in_channels=self.in_channels, out_channels=self.gnn_intermediate_dim, heads=self.gat_heads,
                                            concat=True, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GATv2Conv(in_channels=self.gnn_intermediate_dim * self.gat_heads, out_channels=self.gnn_intermediate_dim,
                                            heads=self.gat_heads, concat=True, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                else:
                    convs.append((GATv2Conv(in_channels=self.gnn_intermediate_dim * self.gat_heads, out_channels=self.gnn_output_node_dim,
                                            heads=self.gat_heads, concat=False, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

            if self.use_vgae:
                self.conv_mu = GATv2Conv(in_channels=self.gnn_output_node_dim * self.gat_heads[1], out_channels=self.gnn_output_node_dim,
                                            heads=self.gat_heads[2], concat=False, dropout=self.gat_dropouts[2])
                self.conv_logstd = GATv2Conv(in_channels=self.gnn_output_node_dim * self.gat_heads[1], out_channels=self.gnn_output_node_dim,
                                            heads=self.gat_heads[3], concat=False, dropout=self.gat_dropouts[3])

        # GIN
        if self.conv_type == 'GIN':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GINConv(
                            torch.nn.Sequential(Linear(in_features=self.in_channels, out_features=self.gnn_intermediate_dim),
                            BatchNorm1d(self.gnn_intermediate_dim),
                            ReLU(),
                            Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                            ReLU()
                           )
                        ), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GINConv(
                            torch.nn.Sequential(Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                            BatchNorm1d(self.gnn_intermediate_dim),
                            ReLU(),
                            Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                            ReLU()
                            )
                        ), 'x, edge_index -> x'))
                else:
                    convs.append((GINConv(
                            torch.nn.Sequential(Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_output_node_dim),
                            BatchNorm1d(self.gnn_output_node_dim),
                            ReLU(),
                            Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                            ReLU()
                            )
                        ), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

            if self.use_vgae:
                self.conv_mu = GINConv(torch.nn.Sequential(
                                    Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                                    BatchNorm1d(self.gnn_output_node_dim),
                                    ReLU(),
                                    Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                                    ReLU()
                                )
                            )
                self.conv_logstd = GINConv(torch.nn.Sequential(
                                    Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                                    BatchNorm1d(self.gnn_output_node_dim),
                                    ReLU(),
                                    Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                                    ReLU()
                                )
                             )

        # PNA
        if self.conv_type == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            deg = get_degrees(self.train_dataset, self.dataset_degree, use_cuda=self.use_cuda)

            pna_common_args = dict(aggregators=aggregators, scalers=scalers, deg=deg,
                                   edge_dim=self.pna_edge_dim, towers=self.pna_num_towers,
                                   pre_layers=self.pna_num_pre_layers, post_layers=self.pna_num_post_layers,
                                   divide_input=False)

            for i in range(self.num_layers):
                if i == 0:
                    convs.append((PNAConv(in_channels=self.in_channels, out_channels=self.gnn_intermediate_dim, **pna_common_args), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((PNAConv(in_channels=self.gnn_intermediate_dim, out_channels=self.gnn_intermediate_dim, **pna_common_args), 'x, edge_index -> x'))
                else:
                    convs.append((PNAConv(in_channels=self.gnn_intermediate_dim, out_channels=self.gnn_output_node_dim, **pna_common_args), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

            if self.use_vgae:
                self.conv_mu = PNAConv(in_channels=self.gnn_output_node_dim, out_channels=self.gnn_output_node_dim, **pna_common_args)
                self.conv_logstd = PNAConv(in_channels=self.gnn_output_node_dim, out_channels=self.gnn_output_node_dim, **pna_common_args)

        self.convs = torch_geometric.nn.Sequential('x, edge_index', convs)


        # Dense readout
        if self.readout == 'dense':
            self.dense_agg = torch.nn.Sequential(
                Linear(in_features=self.max_num_nodes_in_graph * self.gnn_output_node_dim, out_features=self.dense_intermediate_dim),
                BatchNorm1d(self.dense_intermediate_dim),
                ReLU(),
                Linear(in_features=self.dense_intermediate_dim, out_features=self.dense_output_graph_dim),
                BatchNorm1d(self.dense_output_graph_dim),
                ReLU(),
                Dropout(p=0.4)
            )

        # GRU readout
        if self.readout == 'gru':
            self.gru_agg = GRU(input_size=self.max_num_nodes_in_graph, hidden_size=self.gnn_output_node_dim, bidirectional=False, num_layers=1, batch_first=True)

        # Set Transformer readout
        if self.readout == 'set transformer':
            from models.set_transformer_models import SetTransformer
            self.st = SetTransformer(dim_input=self.gnn_output_node_dim, num_outputs=self.set_transformer_k,
                dim_output=self.graph_dim, num_inds=self.set_transformer_num_inds,
                dim_hidden=self.set_transformer_dim_hidden, num_heads=self.set_transformer_num_heads,
                ln=self.set_transformer_layer_norm)

        # Janossy Dense readout
        if self.readout == 'janossy dense':
            from models.janossy_pooling import JanossyPooling
            self.janossy_pool = JanossyPooling(fc_or_rnn='FC', num_perm=self.janossy_num_perms, max_num_atoms_in_graph=self.max_num_nodes_in_graph,
                                               in_features=self.max_num_nodes_in_graph * self.gnn_output_node_dim, fc_intermediate_features=self.dense_intermediate_dim,
                                               fc_out_features=self.graph_dim, use_cuda=self.use_cuda)

        # Janossy GRU readout
        if self.readout == 'janossy gru':
            from models.janossy_pooling import JanossyPooling
            self.janossy_pool = JanossyPooling(fc_or_rnn='GRU', num_perm=self.janossy_num_perms, max_num_atoms_in_graph=self.max_num_nodes_in_graph,
                                               in_features=self.gnn_output_node_dim, use_cuda=self.use_cuda)


        # Regression/classification NN
        in_dim = self.graph_dim

        self.output_nn = torch.nn.Sequential(
            Linear(in_features=in_dim, out_features=self.output_nn_intermediate_dim),
            ReLU(),
            Dropout(p=0.2),
            Linear(in_features=self.output_nn_intermediate_dim, out_features=self.output_nn_out_dim)
        )


    def forward(self, x, edge_index, batch, edge_attr=None):
        # Not using edge features at the moment
        x = x.float()

        x = self.convs(x, edge_index)

        vgae_return = None
        if self.use_vgae:
            mu = self.conv_mu(x, edge_index).relu()
            logstd = self.conv_logstd(x, edge_index).relu()

            logstd = logstd.clamp(max=MAX_LOGSTD)
            # This x would normally be 'z' if purely a VGAE implementation
            x = self._reparametrize(mu, logstd)
            vgae_return = {'mu': mu, 'logstd': logstd}

        if self.readout == 'sum':
            graph_x = global_add_pool(x, batch)

        elif self.readout == 'mean':
            graph_x = global_mean_pool(x, batch)

        elif self.readout == 'max':
            graph_x = global_max_pool(x, batch)

        elif self.readout == 'dense':
            # Regroup nodes into tensor of shape (batch_size, max_num_nodes_in_graph, node_dim)
            graph_x, _ = to_dense_batch(x, batch, fill_value=0, max_num_nodes=self.max_num_nodes_in_graph)
            # Flatten and input to NN
            graph_x = self.dense_agg(graph_x.view(-1, graph_x.shape[1] * graph_x.shape[2]))

        elif self.readout == 'gru':
            graph_x, _ = to_dense_batch(x, batch, fill_value=0, max_num_nodes=self.max_num_nodes_in_graph)
            output, states = self.gru_agg(graph_x.permute(0, 2, 1))
            graph_x = output[:, -1, :]

        elif self.readout == 'set transformer':
            graph_x, _ = to_dense_batch(x, batch, fill_value=0, max_num_nodes=self.max_num_nodes_in_graph)
            graph_x = self.st(graph_x)
            graph_x = graph_x.mean(dim=1)
            # graph_x = graph_x.reshape(-1, self.graph_dim)

        elif self.readout == 'janossy dense' or self.readout == 'janossy gru':
            graph_x, _ = to_dense_batch(x, batch, fill_value=0, max_num_nodes=self.max_num_nodes_in_graph)
            graph_x = self.janossy_pool(graph_x)

        task_predictions = self.output_nn(graph_x)

        return task_predictions, graph_x, x, vgae_return


    def task_loss(self, y_pred, y_true):
        if self.loss_metric == 'BCEWithLogits':
            y_true = y_true.view(y_pred.shape)
            # All labels are 0/1, so binary classification. There might be 1 or more labels.
            task_loss = F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float())

        elif self.loss_metric == 'CrossEntropyLoss':
            task_loss = F.cross_entropy(y_pred.float(), y_true.long())

        elif self.loss_metric == 'MSE':
            y_true = y_true.view(y_pred.shape)
            task_loss = F.mse_loss(y_pred, y_true.float())

        elif self.loss_metric == 'MAE':
            y_true = y_true.view(y_pred.shape)
            task_loss = F.l1_loss(y_pred, y_true.float())

        return task_loss


    ### If using VGAE ###
    def vgae_loss(self, z, mu, logstd, train_pos_edge_index, num_nodes):
        loss = _vgae_recon_loss(z, train_pos_edge_index)
        loss = loss + (1 / num_nodes) * _vgae_kl_loss(mu, logstd)

        return loss

    def _reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    ### If using VGAE ###


    def _step(self, batch, batch_idx):
        x, edge_index, y, batch_ids = batch.x, batch.edge_index, batch.y, batch.batch
        task_predictions, graph_x, x, vgae_return = self.forward(x=x, edge_index=edge_index, batch=batch_ids)

        loss = self.task_loss(task_predictions, y)
        latent_loss = None
        if self.use_vgae:
            latent_loss = self.vgae_loss(x, vgae_return['mu'], vgae_return['logstd'], edge_index, num_nodes=x.shape[0])
            loss += latent_loss

        return loss, latent_loss, y, task_predictions, graph_x


    def training_step(self, batch, batch_idx):
        loss, latent_loss, ys, task_predictions, graph_x = self._step(batch, batch_idx)

        self.log('train_total_loss', loss)
        if latent_loss:
            self.log('train_latent_loss', loss)
        self.train_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': task_predictions})

        return loss


    def validation_step(self, batch, batch_idx):
        loss, latent_loss, ys, task_predictions, graph_x = self._step(batch, batch_idx)

        self.log('validation_total_loss', loss)
        if latent_loss:
            self.log('validation_latent_loss', loss)
        self.validation_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': task_predictions})

        return loss


    def test_step(self, batch, batch_idx):
        loss, latent_loss, ys, task_predictions, graph_x = self._step(batch, batch_idx)

        self.log('test_total_loss', loss)
        if latent_loss:
            self.log('test_latent_loss', loss)
        self.test_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': task_predictions, 'graph_x': graph_x})

        return loss


    def _get_metrics_epoch_end(self, all_y_true, all_y_pred):
        if self.task_type == 'regression':
            all_y_true = all_y_true.detach().cpu().numpy()
            all_y_pred = all_y_pred.detach().cpu().numpy()

            if all_y_true.shape != all_y_pred.shape:
                all_y_pred = all_y_pred.reshape(all_y_true.shape)

            metrics = get_regression_metrics(y_true=all_y_true, y_pred=all_y_pred)
            return metrics

        elif self.task_type == 'binary_classification':
            all_y_pred = torch.sigmoid(all_y_pred)
            all_y_pred = torch.where(all_y_pred >= 0.5, 1.0, 0.0).long()

        elif self.task_type == 'multi_classification':
            all_y_pred_softmax = torch.log_softmax(all_y_pred, dim = 1)
            _, all_y_pred = torch.max(all_y_pred_softmax, dim = 1)

        if 'classification' in self.task_type:
            all_y_pred = all_y_pred.view(all_y_true.shape).squeeze()

        return get_classification_metrics(y_true=all_y_true.long().detach().cpu().numpy(), y_pred=all_y_pred.detach().cpu().numpy())


    def on_train_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.train_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.train_outputs[self.current_epoch]]

        # Uncomment if you want to access the graph embeddings later
        # all_graph_x = [elem['graph_x'] for elem in self.train_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)
        # all_graph_x = torch.cat(all_graph_x, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.train_metrics_per_epoch[self.current_epoch] = metrics
        # self.train_graphs_per_epoch[self.current_epoch] = all_graph_x.detach().cpu().numpy()

        # After saving the metrics delete so that we don't fill the memory
        # Comment if you want to access all the saved data
        del self.train_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred


    def on_validation_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.validation_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.validation_outputs[self.current_epoch]]

        # Uncomment if you want to access the graph embeddings later
        # all_graph_x = [elem['graph_x'] for elem in self.validation_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)
        # all_graph_x = torch.cat(all_graph_x, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.validation_metrics_per_epoch[self.current_epoch] = metrics
        # self.validation_graphs_per_epoch[self.current_epoch] = all_graph_x.detach().cpu().numpy()

        # After saving the metrics delete so that we don't fill the memory
        # Comment if you want to access all the saved data
        del self.validation_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred


    def on_test_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.test_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.test_outputs[self.current_epoch]]
        all_graph_x = [elem['graph_x'] for elem in self.test_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_graph_x = torch.cat(all_graph_x, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.test_metrics_per_epoch[self.current_epoch] = metrics
        self.test_graphs_per_epoch[self.current_epoch] = all_graph_x.detach().cpu().numpy()

#         del self.test_outputs[self.current_epoch]


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('GNN')
        parser.add_argument('--conv_type', type=str)
        parser.add_argument('--gnn_intermediate_dim', type=int)
        parser.add_argument('--gnn_output_node_dim', type=int)
        parser.add_argument('--output_nn_intermediate_dim', type=int)
        parser.add_argument('--readout', type=str)
        parser.add_argument('--learning_rate', type=float)
        parser.add_argument('--use_vgae', dest='use_vgae', action='store_true')
        parser.add_argument('--no-use_vgae', dest='use_vgae', action='store_false')

        # These are optional (depending on the above)
        parser.add_argument('--max_num_nodes_in_graph', type=int)
        parser.add_argument('--gat_heads', type=int)
        parser.add_argument('--gat_dropouts', type=float)
        parser.add_argument('--pna_edge_dim', type=int, default=None, required=False)
        parser.add_argument('--pna_num_towers', type=int)
        parser.add_argument('--pna_num_pre_layers', type=int)
        parser.add_argument('--pna_num_post_layers', type=int)
        parser.add_argument('--dense_intermediate_dim', type=int)
        parser.add_argument('--dense_output_graph_dim', type=int)
        parser.add_argument('--set_transformer_k', type=int)
        parser.add_argument('--set_transformer_dim_hidden', type=int)
        parser.add_argument('--set_transformer_num_heads', type=int)
        parser.add_argument('--set_transformer_layer_norm', dest='set_transformer_layer_norm', action='store_true')
        parser.add_argument('--no-set_transformer_layer_norm', dest='set_transformer_layer_norm', action='store_false')
        parser.add_argument('--set_transformer_num_inds', type=int, default=32)

        return parent_parser
