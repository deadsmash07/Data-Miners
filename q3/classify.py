"""
DO NOT ALTER THIS SCRIPT

Computes the macro averaged ROC-AUC scores by training an SVM over graph features vectors.
"""
from argparse import ArgumentParser
from random import seed

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

SEED = 0
seed(SEED)
np.random.seed(SEED)


# * ----- Parse the arguments.
parser = ArgumentParser()
parser.add_argument("--ftrain", type=str, required=True,
                    help="Path to the training features.")
parser.add_argument("--ftest", type=str, required=True,
                    help="Path to the testing features.")
parser.add_argument("--ltrain", type=str, required=True,
                    help="Path to the training labels.")
parser.add_argument("--ltest", type=str, required=True,
                    help="Path to the testing labels.")
parser.add_argument("--proba", type=str, required=True,
                    help="Path to store the prediction probabilities.")

args = parser.parse_args()


# * ----- Read the features.
features_train = np.load(args.ftrain)
features_test = np.load(args.ftest)


# * ----- Read the labels.
with open(args.ltrain, "r") as file:
    labels_train = file.read()
    labels_train = labels_train.strip().split("\n")
    labels_train = [int(i) for i in labels_train]
labels_train = np.array(labels_train, dtype=int)

with open(args.ltest, "r") as file:
    labels_test = file.read()
    labels_test = labels_test.strip().split("\n")
    labels_test = [int(i) for i in labels_test]
labels_test = np.array(labels_test, dtype=int)


# * ----- Train and test the model.
model = SVC(kernel="rbf", class_weight="balanced",
            max_iter=10000, probability=True, random_state=SEED)
model.fit(features_train, labels_train)

# Probability of the class 1.
proba_train = model.predict_proba(features_train)[:, 1]
roc_auc_train = roc_auc_score(
    y_true=labels_train, y_score=proba_train, average="macro")

proba_test = model.predict_proba(features_test)[:, 1]
roc_auc_test = roc_auc_score(
    y_true=labels_test, y_score=proba_test, average="macro")

print(f"Train ROC_AUC: {roc_auc_train:.3f}")
print(f"Test ROC_AUC: {roc_auc_test:.3f}")


# * ----- Save the probabilities.
np.save(args.proba, proba_test)
