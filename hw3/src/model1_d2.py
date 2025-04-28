import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv
from torch_geometric.nn import global_mean_pool

class GCN1(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.5, heads=4):
        super().__init__()
        # First GCN layer
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.bn1 = BatchNorm(hidden_feats)
        
        # GAT layer for attention mechanism
        self.gat = GATConv(hidden_feats, hidden_feats, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_feats * heads)
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_feats * heads, hidden_feats)
        self.bn3 = BatchNorm(hidden_feats)
        
        # Final classification layer
        self.lin = torch.nn.Linear(hidden_feats, num_classes)
        
        self.dropout = dropout
        self.hidden_feats = hidden_feats
        self.heads = heads

    def forward(self, x, edge_index, batch=None):
        # First GCN layer
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # GAT layer with attention
        x2 = F.elu(self.gat(x1, edge_index))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Second GCN layer with skip connection
        x3 = F.relu(self.bn3(self.conv2(x2, edge_index)))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x3 = x3 + x1  # Skip connection
        
        # Global pooling if batch is provided
        if batch is not None:
            x3 = global_mean_pool(x3, batch)
        
        # Final classification
        x = self.lin(x3)
        return F.log_softmax(x, dim=1)
