# eval_task1_d1.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 1) load ground truth
y = np.load('hw3_col761_2025_datasets/datasets/task1/d1/label.npy')
mask = ~np.isnan(y)
y_true = y[mask].astype(int)

# 2) load your predicted probabilities (n × 5)
probs = pd.read_csv('preds1_d1.csv', header=None).values

# 3) compute multiclass ROC‑AUC (one‑vs‑rest, macro‐averaged)
auc = roc_auc_score(
    y_true, 
    probs[mask], 
    multi_class='ovr',    # or 'ovo'
    average='macro'
)
print(f'ROC‑AUC (d1): {auc:.4f}')
