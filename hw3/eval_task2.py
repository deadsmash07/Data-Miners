import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import argparse

def calculate_weighted_f1(true_labels_path, predicted_labels_path):
    # Read the true labels from .npy file and predicted labels from CSV
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
    parser.add_argument('--pred', required=True, help='Path to predicted labels CSV file')
    args = parser.parse_args()
    true_labels_path='datasets/task2/train/label.npy'
    calculate_weighted_f1(true_labels_path, args.pred) 

# python eval_task2.py --true path/to/true_labels.npy --pred path/to/predicted_labels.csv