import torch
import random
import pytorch_lightning as pl
from torch.nn import GRU


class JanossyPooling(pl.LightningModule):
    def __init__(self,
                 fc_or_rnn: str,
                 num_perm: int,
                 max_num_atoms_in_graph: int,
                 in_features: int,
                 fc_intermediate_features: int = None,
                 fc_out_features: int = None,
                 out_features: int = None,
                 use_cuda: bool = True,
                 ):
        super(JanossyPooling, self).__init__()

        self.num_perm = num_perm
        self.fc_or_rnn = fc_or_rnn
        self.use_cuda = use_cuda

        if self.fc_or_rnn == 'FC':
            self.nn = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=fc_intermediate_features),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=fc_intermediate_features, out_features=fc_out_features),
                torch.nn.ReLU()
            )

        elif self.fc_or_rnn == 'GRU':
            self.nn = GRU(batch_first=True, input_size=max_num_atoms_in_graph, hidden_size=in_features, bidirectional=False, num_layers=1)

        # self.rho_function = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=fc_out_features, out_features=fc_out_features // 2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=fc_out_features // 2, out_features=out_features)
        # )


    def forward(self, x):
        # We assume the input is structured as follows
        num_batches = x.shape[0]
        num_nodes = x.shape[1]
        num_features = x.shape[2]

        # x with permutations has shape (num_perm, batch_size, max_num_nodes_in_graph, num_features)
        x_permutations = JanossyPooling._tensor_permutation(x, self.num_perm, use_cuda=self.use_cuda)

        # First reshape to think of additional permutations as a larger batch
        x_permutations = x_permutations.view(num_batches * self.num_perm, *x_permutations.shape[2:])

        if self.fc_or_rnn == 'FC':
            # Now flatten for the fully connected NN
            x_permutations = x_permutations.view(-1, num_nodes * num_features)

            # out has shape (num_perm * batch_size, fc_out_features)
            out = self.nn(x_permutations)

            # Rearrange out to recover permutations; final shape (num_perm, batch_size, fc_out_features)
            out = torch.stack(torch.tensor_split(out, self.num_perm, dim=0))

        elif self.fc_or_rnn == 'GRU':
            output, states = self.nn(x_permutations.permute(0, 2, 1))

            if self.fc_or_rnn == 'GRU':
                h_n = states
            else:
                (h_n, c_n) = states

            # # Final hidden state for the last character
            # out = torch.flatten(h_n)

            # output should have shape (batch_size, max_num_nodes_in_graph, lstm_out)
            # Features for the last input
            out = output[:, -1, :]

            # Rearrange out to recover permutations; final shape (num_perm, batch_size, ???)
            out = torch.stack(torch.tensor_split(out, self.num_perm, dim=0))


        # According to Janossy pooling, now need to take mean across permutations
        out = out.mean(dim=0)

        # Apply second function (rho)
        # return self.rho_function(out)

        return out


    @staticmethod
    def _tensor_permutation(tensor, num_perm, use_cuda):
        num_batches = tensor.shape[0]
        num_nodes = tensor.shape[1]
        num_features = tensor.shape[2]

        random_indices = torch.stack([torch.stack([torch.stack([torch.Tensor(random.sample(range(num_features), num_features)) for _ in range(num_nodes)]) for _ in range(num_batches)]) for _ in range(num_perm)]).long()
        random_indices = random_indices.cuda() if use_cuda else random_indices
        tensor_repeated = tensor.reshape(1, *tensor.shape).repeat(num_perm, 1, 1, 1)

        return torch.gather(tensor_repeated, 3, random_indices)
