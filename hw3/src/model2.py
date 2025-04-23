import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

class BipartiteGNN(torch.nn.Module):
    def __init__(self, d_u, d_p, hidden_dim=128, out_dim=1, dropout=0.3):
        super().__init__()
        # initial projections
        self.lin_u    = torch.nn.Linear(d_u, hidden_dim)
        self.lin_p    = torch.nn.Linear(d_p, hidden_dim)

        # two GraphSAGE layers with batch‐norm
        self.conv1    = SAGEConv(hidden_dim, hidden_dim)
        self.bn1      = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2    = SAGEConv(hidden_dim, hidden_dim)
        self.bn2      = torch.nn.BatchNorm1d(hidden_dim)

        # final MLP head
        self.lin_out  = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout  = dropout

    def forward(self, x_u, x_p, edge_index):
        # 1) Project and stack user + product features
        x_u0 = F.relu(self.lin_u(x_u))
        x_p0 = F.relu(self.lin_p(x_p))
        x   = torch.cat([x_u0, x_p0], dim=0)

        # 2) Make edges undirected
        edge_index = to_undirected(edge_index)

        # 3) First GraphSAGE + batchnorm + activation + dropout
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.dropout, training=self.training)

        # 4) Second GraphSAGE + batchnorm + activation
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)

        # 5) Residual: add initial embedding back
        h2 = h2 + x

        # 6) Only return user‐node embeddings, then final MLP
        out = self.lin_out(h2[: x_u.size(0)])
        return out  # raw logits; apply sigmoid at inference
