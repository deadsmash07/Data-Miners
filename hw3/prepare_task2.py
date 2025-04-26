#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Task2 data: split into train and test PT files.'
    )
    parser.add_argument(
        '--data_dir', '-d', type=str, required=True,
        help='Directory containing train/ subfolder with .npy files.'
    )
    parser.add_argument(
        '--train-output', '-tr', type=str, required=True,
        help='Output .pt filename for training data.'
    )
    parser.add_argument(
        '--test-output', '-te', type=str, required=True,
        help='Output .pt filename for test data.'
    )
    parser.add_argument(
        '--split', '-s', type=float, default=0.8,
        help='Fraction of labeled data to use for training (default: 0.8).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible split.'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    train_output = args.train_output
    test_output = args.test_output
    split_ratio = args.split
    seed = args.seed

    # Load raw data
    train_dir = os.path.join(data_dir, 'train')
    uf_path = os.path.join(train_dir, 'user_features.npy')
    pf_path = os.path.join(train_dir, 'product_features.npy')
    up_path = os.path.join(train_dir, 'user_product.npy')
    lbl_path = os.path.join(train_dir, 'label.npy')

    print(f"Loading user features from {uf_path}")
    user_features = np.load(uf_path)
    print(f"Loading product features from {pf_path}")
    product_features = np.load(pf_path)
    print(f"Loading edge list from {up_path}")
    user_product = np.load(up_path)
    print(f"Loading labels from {lbl_path}")
    labels = np.load(lbl_path)

    n_users = user_features.shape[0]
    n_products = product_features.shape[0]

    # Identify users with labels (remove graphs with any NaN labels)
    labeled_mask = ~np.isnan(labels).any(axis=1)
    labeled_indices = np.where(labeled_mask)[0]
    n_labeled = labeled_indices.shape[0]
    print(f"Total users: {n_users}, labeled users: {n_labeled}, removed {n_users - n_labeled} unlabeled users.")

    # Split labeled users into train and test
    np.random.seed(seed)
    perm = np.random.permutation(n_labeled)
    n_train = int(split_ratio * n_labeled)
    train_idx = labeled_indices[perm[:n_train]]
    test_idx = labeled_indices[perm[n_train:]]
    n_test = test_idx.shape[0]
    print(f"Training on {n_train} users, testing on {n_test} users.")

    # Prepare train split
    uf_train = user_features[train_idx]
    y_train = labels[train_idx]
    # Filter edges for train users
    mask_edges_train = np.isin(user_product[:, 0], train_idx)
    edges_train = user_product[mask_edges_train].copy()
    # Reindex user IDs
    user_old2new_train = {old: new for new, old in enumerate(train_idx)}
    edges_train[:, 0] = [user_old2new_train[old] for old in edges_train[:, 0]]
    # Reindex product IDs: subtract original user count, then offset by new train user count
    edges_train[:, 1] = edges_train[:, 1] - n_users + n_train

    # Prepare test split
    uf_test = user_features[test_idx]
    y_test = labels[test_idx]
    mask_edges_test = np.isin(user_product[:, 0], test_idx)
    edges_test = user_product[mask_edges_test].copy()
    user_old2new_test = {old: new for new, old in enumerate(test_idx)}
    edges_test[:, 0] = [user_old2new_test[old] for old in edges_test[:, 0]]
    edges_test[:, 1] = edges_test[:, 1] - n_users + n_test

    # Convert to torch tensors
    train_data = {
        'user_features': torch.from_numpy(uf_train).float(),
        'product_features': torch.from_numpy(product_features).float(),
        'user_product': torch.from_numpy(edges_train).long(),
        'labels': torch.from_numpy(y_train).long()
    }
    test_data = {
        'user_features': torch.from_numpy(uf_test).float(),
        'product_features': torch.from_numpy(product_features).float(),
        'user_product': torch.from_numpy(edges_test).long(),
        'labels': torch.from_numpy(y_test).long()
    }

    # Save processed splits
    print(f"Saving training data to {train_output}")
    torch.save(train_data, train_output)
    print(f"Saving test data to {test_output}")
    torch.save(test_data, test_output)
    print("Data preparation complete.")


if __name__ == '__main__':
    main() 