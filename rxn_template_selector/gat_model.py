from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from functools import reduce

class GATModel(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        # self.layers = map(lambda out, heads, dropout: gnn.GATConv(-1, out, heads=heads, dropout=dropout), zip(layers))
        self.conv1 = gnn.GATConv(-1, 64, heads=8, dropout=0.1)
        self.conv2 = gnn.GATConv(-1, 256, heads=8, dropout=0.1)
        self.conv3 = gnn.GATConv(-1, 256, heads=8, dropout=0.1)
        self.conv4 = gnn.GATConv(-1, 256, heads=8, dropout=0.1)
        self.conv_final = gnn.GATConv(-1, 256, heads=8, dropout=0.1, concat=False)

        # self.pool1 = gnn.GlobalAttention(nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ELU(),
        #     nn.Linear(256, 1),
        # ))
        self.pool1 = gnn.global_mean_pool

        self.lin1 = nn.Linear(256, output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv_final(x, edge_index))

        x = F.elu(self.pool1(x, batch))
        x = self.lin1(x)

        return x

    def pipe_through(self, x, edge_index):
        return reduce(lambda x, layer: layer(x, edge_index), self.layers, x)
