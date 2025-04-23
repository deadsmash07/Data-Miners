import argparse
import numpy as np
import torch
from torch_geometric.data import Data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edges',    required=True, help='E×2 numpy of edge list')
    p.add_argument('--features', required=True, help='n×d node features')
    p.add_argument('--labels',   required=True, help='n labels (int or nan)')
    p.add_argument('--output',   required=True, help='where to write out the .pt')
    args = p.parse_args()

    edges = np.load(args.edges)         # shape [E,2]
    x     = np.load(args.features)      # shape [n,d]
    y     = np.load(args.labels)        # shape [n,] with nan for test

    # to torch
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x_t        = torch.tensor(x, dtype=torch.float)
    y_t        = torch.tensor(y, dtype=torch.float)  # float so we can carry nan

    data = Data(x=x_t, edge_index=edge_index, y=y_t)
    torch.save(data, args.output)

if __name__=='__main__':
    main()
