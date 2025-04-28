#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Task-2 data: filter out users with NaN labels and save a single .pt file."
    )
    parser.add_argument(
        "--data_dir", "-d", required=True,
        help="Directory containing a train/ subfolder with .npy files."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Where to write the cleaned graph (.pt)."
    )
    args = parser.parse_args()

    # ── 1. load raw arrays ────────────────────────────────────────────────────
    # train_dir = os.path.join(args.data_dir, "train")
    train_dir = args.data_dir
    uf_path   = os.path.join(train_dir, "user_features.npy")
    pf_path   = os.path.join(train_dir, "product_features.npy")
    up_path   = os.path.join(train_dir, "user_product.npy")
    lbl_path  = os.path.join(train_dir, "label.npy")

    print("Loading …")
    user_features   = np.load(uf_path)
    product_features = np.load(pf_path)
    user_product    = np.load(up_path)
    labels          = np.load(lbl_path)

    n_users, n_products = user_features.shape[0], product_features.shape[0]

    # ── 2. drop users whose label has any NaN ─────────────────────────────────
    keep_mask    = ~np.isnan(labels).any(axis=1)     # shape [n_users]
    keep_indices = np.nonzero(keep_mask)[0]          # old user-ids to keep
    n_keep       = keep_indices.size
    print(f"Total users: {n_users} → keeping {n_keep} (removed {n_users - n_keep}).")

    # prune user features & labels
    uf_clean = user_features[keep_mask]
    y_clean  = labels[keep_mask]

    # ── 3. prune + re-index edge list ─────────────────────────────────────────
    # keep an edge only if its user endpoint survived
    edge_mask        = np.isin(user_product[:, 0], keep_indices)
    edges_kept       = user_product[edge_mask].copy()

    # map old user-id → new consecutive id 0 … n_keep-1
    user_old2new = {old: new for new, old in enumerate(keep_indices)}
    edges_kept[:, 0] = [user_old2new[old] for old in edges_kept[:, 0]]

    # re-index products so they follow the user indices contiguously
    #   original product ids were  n_users … n_users+n_products-1
    #   new product ids will be     n_keep … n_keep+n_products-1
    edges_kept[:, 1] = edges_kept[:, 1] - n_users + n_keep

    # ── 4. package into a dict of torch tensors ───────────────────────────────
    graph = {
        "user_features"   : torch.from_numpy(uf_clean).float(),
        "product_features": torch.from_numpy(product_features).float(),
        "user_product"    : torch.from_numpy(edges_kept).long(),
        "labels"          : torch.from_numpy(y_clean).long(),
    }

    # ── 5. save ───────────────────────────────────────────────────────────────
    torch.save(graph, args.output)
    print(f"Cleaned graph saved to {args.output}")


if __name__ == "__main__":
    main()
