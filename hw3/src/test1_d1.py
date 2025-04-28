import argparse
import torch
import pandas as pd
from model1_d1 import ImprovedGNN

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.input, map_location=device)
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    # load model
    # Calculate number of classes using a different approach
    y_without_nan = data.y[~torch.isnan(data.y)]
    if len(y_without_nan) > 0:
        num_classes = int(torch.max(y_without_nan).item() + 1)
    else:
        # If all values are NaN, you might need to specify a default value
        # or infer it from another source like the model checkpoint
        num_classes = 2  # Default value, adjust based on your dataset

    model = ImprovedGNN(
        in_feats=x.size(1), 
        hidden_feats=128, 
        num_classes=num_classes,
        dropout=0.3,
        num_layers=4
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # predict
    with torch.no_grad():
        logits = model(x, edge_index)
        probs  = torch.softmax(logits, dim=1)

    # -- write out the full probability matrix [n_nodes × num_classes]
    pd.DataFrame(probs.cpu().numpy()).to_csv(
        args.output, header=False, index=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='path to test graph (.pt)')
    parser.add_argument('--model',  required=True, help='path to saved model.pt')
    parser.add_argument('--output', required=True, help='csv to save predictions')
    args = parser.parse_args()
    main(args)