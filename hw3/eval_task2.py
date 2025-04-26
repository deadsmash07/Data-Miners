import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import argparse
import torch

def calculate_weighted_f1(true_labels_path, predicted_labels_path):
    # Load true labels (support .npy and .pt) and predicted labels from CSV
    if true_labels_path.endswith('.pt'):
        data = torch.load(true_labels_path, map_location='cpu')
        true_labels = data['labels'].cpu().numpy()
    else:
        true_labels = np.load(true_labels_path)
    predicted_labels = pd.read_csv(predicted_labels_path, header=None).values
    
    # Flatten both arrays for F1 score computation
    true_labels_flat = true_labels.reshape(-1)
    predicted_labels_flat = predicted_labels.reshape(-1)
    
    # Convert probabilities to binary predictions (threshold at 0.5)
    predicted_binary = (predicted_labels_flat >= 0.5).astype(int)
    # for i in range(len(predicted_binary)):
    #     if predicted_binary[i] == 0:
    #         print(predicted_labels_flat[i])
    # Calculate weighted F1 score using flattened vectors
    f1 = f1_score(true_labels_flat, predicted_binary, average='weighted')
    
    print(f"Weighted F1 Score: {f1:.4f}")
    return f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate weighted F1 score')
    parser.add_argument('--true', required=True, help='Path to true labels file (.npy or .pt)')
    parser.add_argument('--pred', required=True, help='Path to predicted labels CSV file')
    args = parser.parse_args()
    calculate_weighted_f1(args.true, args.pred)

# python eval_task2.py --true path/to/true_labels.npy --pred path/to/predicted_labels.csv