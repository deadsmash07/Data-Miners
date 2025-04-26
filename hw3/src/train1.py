import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from model1 import GCN1
import numpy as np

def main(args):
    
    # Set device with explicit CUDA visibility
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    # Load data to CPU first
    print(f"Loading data from {args.input}...")
    data = torch.load(args.input, map_location='cpu')
    x, edge_index, y = data.x, data.edge_index, data.y
    train_mask = data.train_mask

    # Move data to device
    x, edge_index = x.to(device), edge_index.to(device)
    y = y.to(device)
    
    # Get the maximum class index
    max_class = int(torch.max(y[train_mask]).item())

    # model
    model = GCN1(in_feats=x.size(1),
                 hidden_feats=64,
                 num_classes=max_class+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    model.train()
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        out = model(x, edge_index)
        
        # Training loss
        train_loss = F.cross_entropy(out[train_mask], y[train_mask].long())
        
        train_loss.backward()
        optimizer.step()
        scheduler.step(train_loss)

        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss.item():.4f}')      

    # save
    torch.save(model.state_dict(), args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to train graph (.pt)')
    parser.add_argument('--output', required=True, help='where to save model.pt')
    args = parser.parse_args()
    main(args)