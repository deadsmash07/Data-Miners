import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BipartiteGNN(torch.nn.Module):
    def __init__(self, d_u, d_p, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        # project users and products to common space
        self.lin_u = torch.nn.Linear(d_u, hidden_dim)
        self.lin_p = torch.nn.Linear(d_p, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin_out = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x_u, x_p, edge_index):
        # stack
        x = torch.cat([
            F.relu(self.lin_u(x_u)),
            F.relu(self.lin_p(x_p))
        ], dim=0)
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # only return user‐node outputs
        return self.lin_out(x[:x_u.size(0)])  # [m × ℓ]
