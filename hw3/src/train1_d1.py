#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from model1_d1 import ImprovedGNN


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
    
    model = ImprovedGNN(
        in_feats=x.size(1),
        hidden_feats=128,
        num_classes=num_classes,
        dropout=0.3,
        num_layers=4
    ).to(device)
    
    # Improved optimizer with different weight decay for different parameter groups
    no_decay = ['bias', 'BatchNorm']
    params = [
        {'params': [p for n, p in model.named_parameters() 
                   if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters() 
                   if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(params, lr=0.005)
    
    # Learning rate scheduler with warmup
    max_epochs = 701
    warm_up_steps = 50
    def lr_lambda(epoch):
        if epoch < warm_up_steps:
            # Linear warmup
            return float(epoch) / float(max(1, warm_up_steps))
        # Cosine annealing
        return 0.5 * (1 + np.cos(np.pi * float(epoch - warm_up_steps) / 
                                 float(max_epochs - warm_up_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── training loop ────────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, max_epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)

        loss = F.cross_entropy(out[labeled_mask], y[labeled_mask].long())
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}  •  loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── save ─────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to graph (.pt)')
    parser.add_argument('--output', required=True, help='where to save model weights (.pt)')
    main(parser.parse_args())
