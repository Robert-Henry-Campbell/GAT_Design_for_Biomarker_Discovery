import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear
from sag_pool_custom import SAGPoolingCustom


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, pooling_keep_count, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.pool1 = SAGPoolingCustom(hidden_channels * num_heads, min_score=None,
                                      multiplier=1.0, ratio=pooling_keep_count)
        self.lin = Linear(hidden_channels * num_heads, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * num_heads)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, perm, score, pre_drop_score_two = self.pool1(
            x, edge_index, None, batch
        )
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        pre_drop_score = pre_drop_score_two
        return x, pre_drop_score
