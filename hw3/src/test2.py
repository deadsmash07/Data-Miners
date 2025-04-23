import argparse
import torch
import pandas as pd
from model2 import BipartiteGNN

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.input, map_location=device)
    
    x_u, x_p = data['user_features'].to(device), data['product_features'].to(device)
    edge_index = data['user_product'].to(device)
    y_u = data['labels'].to(device)  # [m × ℓ] floats (0/1), nan for test rows

    # Ensure edge_index is in the correct format (2 x num_edges)
    if edge_index.dim() == 1:
        edge_index = edge_index.unsqueeze(0)
    if edge_index.size(0) != 2:
        edge_index = edge_index.t()

    # load
    model = BipartiteGNN(d_u=x_u.size(1),
                         d_p=x_p.size(1),
                         hidden_dim=128,
                         out_dim=y_u.size(1),dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x_u, x_p, edge_index)
        probs  = torch.sigmoid(logits).cpu().numpy()

    # save (mt × ℓ)
    pd.DataFrame(probs).to_csv(args.output, header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to test graph (.pt)')
    parser.add_argument('--model',  required=True, help='path to saved model.pt')
    parser.add_argument('--output', required=True, help='csv to save predictions')
    args = parser.parse_args()
    main(args)
