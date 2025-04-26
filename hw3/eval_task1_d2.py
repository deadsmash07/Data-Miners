# eval_task1_d2.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 1) load ground truth
y = np.load('datasets/task1/d2/label.npy')
mask = ~np.isnan(y)
y_true = y[mask].astype(int)

# 2) load your predicted labels
y_pred = pd.read_csv('preds1_d2.csv', header=None).values[mask].flatten()

# 3) compute accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy (d2): {accuracy:.4f}')
