#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from model1_d2 import GCN1


def main(args):
    # ── device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")

    # ── load graph ────────────────────────────────────────────────────────────
    print(f"Loading data from {args.input} …")
    data: Data = torch.load(args.input, map_location='cpu')

    x, edge_index, y = data.x, data.edge_index, data.y
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

    # ── keep only labeled nodes (y not NaN) ──────────────────────────────────
    labeled_mask = ~torch.isnan(y)
    if labeled_mask.sum() == 0:
        raise ValueError("No labeled nodes found (all y are NaN).")

    # ── model & optimizer ────────────────────────────────────────────────────
    num_classes = int(torch.max(y[labeled_mask]).item()) + 1
    model = GCN1(in_feats=x.size(1), hidden_feats=64, num_classes=num_classes).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # ── training loop ────────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        out = model(x, edge_index)

        loss = F.cross_entropy(out[labeled_mask], y[labeled_mask].long())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}  •  loss: {loss.item():.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to graph (.pt)')
    parser.add_argument('--output', required=True, help='where to save model weights (.pt)')
    main(parser.parse_args())
