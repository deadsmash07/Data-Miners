import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN1(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits
