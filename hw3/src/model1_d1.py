import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm, global_mean_pool, SAGEConv, JumpingKnowledge
import torch.nn as nn

class ImprovedGNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats=128, num_classes=2, dropout=0.5, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection layer
        self.input_lin = nn.Linear(in_feats, hidden_feats)
        
        # Multiple GNN layers with different architectures
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer (GIN)
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.BatchNorm1d(hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats),
                nn.BatchNorm1d(hidden_feats),
                nn.ReLU()
            ), train_eps=True)
        )
        self.batch_norms.append(BatchNorm(hidden_feats))
        
        # Middle layers (GraphSAGE with residual connections)
        for i in range(1, num_layers-1):
            self.convs.append(SAGEConv(hidden_feats, hidden_feats))
            self.batch_norms.append(BatchNorm(hidden_feats))
        
        # Final layer (GIN again)
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.BatchNorm1d(hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats),
                nn.BatchNorm1d(hidden_feats),
                nn.ReLU()
            ), train_eps=True)
        )
        self.batch_norms.append(BatchNorm(hidden_feats))
        
        # Jumping Knowledge to combine features from all layers
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_feats, num_layers=num_layers)
        
        # Output layer
        self.lin1 = nn.Linear(num_layers * hidden_feats, hidden_feats)
        self.lin2 = nn.Linear(hidden_feats, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        # Initial feature transformation
        x = self.input_lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store all layer representations for JK
        xs = []
        
        # Process through GNN layers
        for i in range(self.num_layers):
            # Apply convolution
            x_conv = self.convs[i](x, edge_index)
            x_conv = self.batch_norms[i](x_conv)
            x_conv = F.relu(x_conv)
            
            # Residual connection for all but first layer
            if i > 0:
                x = x_conv + x  # Residual connection
            else:
                x = x_conv
                
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Store layer representation
            xs.append(x)
        
        # Combine features from all layers using JK
        x = self.jk(xs)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)

# Keep GINNet for backward compatibility
class GINNet(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.5):
        super().__init__()
        # Two-layer GIN
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats)
        ))
        self.bn1 = BatchNorm(hidden_feats)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats)
        ))
        self.bn2 = BatchNorm(hidden_feats)
        # Final classification
        self.lin = torch.nn.Linear(hidden_feats, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # First GIN layer
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second GIN layer
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        # Global pooling for graph-level readout
        if batch is not None:
            x = global_mean_pool(x, batch)
        # Classification head
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
