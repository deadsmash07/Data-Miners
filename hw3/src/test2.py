import argparse
import torch
import pandas as pd
from model2 import BipartiteGNN

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.input, map_location=device)
    x_u, x_p = data.x_u.to(device), data.x_p.to(device)
    edge_index = data.edge_index.to(device)

    # load
    model = BipartiteGNN(d_u=x_u.size(1),
                         d_p=x_p.size(1),
                         hidden_dim=64,
                         out_dim=data.y_u.size(1)).to(device)
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
