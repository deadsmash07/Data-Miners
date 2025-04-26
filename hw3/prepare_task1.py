import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edges',    required=True, help='E×2 numpy of edge list')
    p.add_argument('--features', required=True, help='n×d node features')
    p.add_argument('--labels',   required=True, help='n labels (int or nan)')
    p.add_argument('--train-output', required=True, help='where to write out the train .pt')
    p.add_argument('--test-output', required=True, help='where to write out the test .pt')
    p.add_argument('--test-size', type=float, default=0.2, help='fraction of data to use for testing')
    p.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = p.parse_args()

    # Load data
    edges = np.load(args.edges)         # shape [E,2]
    x     = np.load(args.features)      # shape [n,d]
    y     = np.load(args.labels)        # shape [n,] with nan for test

    # Convert to torch tensors
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x_t        = torch.tensor(x, dtype=torch.float)
    y_t        = torch.tensor(y, dtype=torch.float)  # float so we can carry nan

    # Create train/test masks
    # Only split nodes that have labels (not nan)
    labeled_mask = ~torch.isnan(y_t)
    labeled_indices = torch.where(labeled_mask)[0]
    
    # Split indices into train and test
    train_idx, test_idx = train_test_split(
        labeled_indices.numpy(),
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True
    )
    
    # Create train and test masks
    train_mask = torch.zeros_like(y_t, dtype=torch.bool)
    test_mask = torch.zeros_like(y_t, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Create train data
    train_data = Data(
        x=x_t,
        edge_index=edge_index,
        y=y_t,
        train_mask=train_mask,
        test_mask=test_mask
    )

    # Save train and test data
    torch.save(train_data, args.train_output)
    print(f"Saved train data to {args.train_output}")
    print(f"Train nodes: {train_mask.sum().item()}, Test nodes: {test_mask.sum().item()}")

if __name__=='__main__':
    main()
