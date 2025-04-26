import argparse
import torch
import torch.nn.functional as F
from model2 import BipartiteGNN
from tqdm import tqdm
def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    data = torch.load(args.input, map_location=device)
    # x_u, x_p = data.x_u.to(device), data.x_p.to(device)
    # edge_index    = data.edge_index.to(device)
    # y_u           = data.y_u.to(device)  # [m × ℓ] floats (0/1), nan for test rows
    x_u, x_p = data['user_features'].to(device), data['product_features'].to(device)
    edge_index = data['user_product'].to(device)
    y_u = data['labels'].to(device)  # [m × ℓ] floats (0/1), nan for test rows

    # Debug prints
    print(f"x_u shape: {x_u.shape}, dtype: {x_u.dtype}")
    print(f"x_p shape: {x_p.shape}, dtype: {x_p.dtype}")
    print(f"edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
    print(f"y_u shape: {y_u.shape}, dtype: {y_u.dtype}")

    # Ensure edge_index is in the correct format (2 x num_edges)
    if edge_index.size(0) != 2:
        edge_index = edge_index.t()

    # Ensure edge_index is long type (required for PyTorch Geometric)
    edge_index = edge_index.long()

    print(f"edge_index shape after processing: {edge_index.shape}, dtype: {edge_index.dtype}")

    # mask train users
    train_mask = ~torch.isnan(y_u).any(dim=1)
    y_train = y_u[train_mask]

    # Ensure y_train is float type for BCEWithLogitsLoss
    y_train = y_train.float()

    model = BipartiteGNN(d_u=x_u.size(1),
                         d_p=x_p.size(1),
                         hidden_dim=128,
                         out_dim=y_u.size(1),dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    pbar = tqdm(range(1, 1001), desc='Training', unit='epoch')
    for epoch in pbar:
        optimizer.zero_grad()
        out = model(x_u, x_p, edge_index)
        loss = criterion(out[train_mask], y_train)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to train graph (.pt)')
    parser.add_argument('--output', required=True, help='where to save model.pt')
    args = parser.parse_args()
    main(args)
