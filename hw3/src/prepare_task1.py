#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edges',    required=True, help='E×2 numpy file of edge list')
    p.add_argument('--features', required=True, help='n×d numpy file of node features')
    p.add_argument('--labels',   required=True, help='n-length numpy file of labels (int or nan)')
    p.add_argument('--output',   required=True, help='path to write the combined .pt file')
    args = p.parse_args()

    # ----- load numpy arrays --------------------------------------------------
    edges_np = np.load(args.edges)        # shape [E, 2]
    x_np     = np.load(args.features)     # shape [n, d]
    y_np     = np.load(args.labels)       # shape [n]

    # ----- convert to torch tensors ------------------------------------------
    edge_index = torch.tensor(edges_np.T, dtype=torch.long)   # shape [2, E]
    x          = torch.tensor(x_np,        dtype=torch.float) # shape [n, d]
    y          = torch.tensor(y_np,        dtype=torch.float) # keep float so NaN is preserved

    # ----- build Data object --------------------------------------------------
    data = Data(x=x, edge_index=edge_index, y=y)

    # ----- save ---------------------------------------------------------------
    torch.save(data, args.output)
    print(f"Saved full dataset to {args.output}")
    print(f"Nodes: {x.size(0)}, Edges: {edge_index.size(1)}")

if __name__ == "__main__":
    main()
