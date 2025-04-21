import argparse
import torch
import torch.nn.functional as F
from model2 import BipartiteGNN

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.input, map_location=device)
    x_u, x_p = data.x_u.to(device), data.x_p.to(device)
    edge_index    = data.edge_index.to(device)
    y_u           = data.y_u.to(device)  # [m × ℓ] floats (0/1), nan for test rows

    # mask train users
    train_mask = ~torch.isnan(y_u).any(dim=1)
    y_train    = y_u[train_mask]

    model = BipartiteGNN(d_u=x_u.size(1),
                         d_p=x_p.size(1),
                         hidden_dim=64,
                         out_dim=y_u.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(x_u, x_p, edge_index)
        loss = criterion(out[train_mask], y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to train graph (.pt)')
    parser.add_argument('--output', required=True, help='where to save model.pt')
    args = parser.parse_args()
    main(args)
