import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from model1 import GCN1

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.input, map_location=device)  # expects Data.x, Data.edge_index, Data.y
    x, edge_index, y = data.x, data.edge_index, data.y

    # mask: only train on non-nan labels
    train_mask = ~torch.isnan(y)
    y = y[train_mask].long()
    x, edge_index = x.to(device), edge_index.to(device)
    
    # Get the maximum class index, replacing torch.nanmax which isn't available
    max_class = int(torch.max(y).item())

    # model
    model = GCN1(in_feats=x.size(1),
                 hidden_feats=64,
                 num_classes=max_class+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    # save
    torch.save(model.state_dict(), args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to train graph (.pt)')
    parser.add_argument('--output', required=True, help='where to save model.pt')
    args = parser.parse_args()
    main(args)