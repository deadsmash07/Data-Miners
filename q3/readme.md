- for preprocesing:(everything in q3 directory)
activate virtual env:
```bash
source dm/bin/activate
```

Identify your subgraphs using training data only. For instance:

```bash 
identify.sh q3_datasets/Mutagenicity/train_preprocessed_gspan.dat q3_datasets/Mutagenicity/train_labels.txt /path/to/gspan 0.005
```

This step calls subgraph_mining.py --mode identify on the training subset. gSpan writes the frequent patterns to train_preprocessed_gspan.dat.fp, and the script prints the top subgraphs by chi-square (no separate subgraph file is created).

Convert the training graphs to a feature matrix:

```bash 
convert.sh q3_datasets/Mutagenicity/train_preprocessed_gspan.dat output/mutag_features_train.npy
```
This creates output/mutag_features_train.npy (shape [#train_graphs, #subgraphs]).

Convert the test graphs to a feature matrix (get the test script from the data-split have a separate test_preprocessed_gspan.dat or the TAs generate one):

```bash 
convert.sh q3_datasets/Mutagenicity/test_preprocessed_gspan.dat output/mutag_features_test.npy
```
This creates output/mutag_features_test.npy.

    Note: The .fp file for test_preprocessed_gspan.dat must exist or the script will parse that file’s .fp for membership sets. This only works if the test set is labeled exactly the same way as gSpan’s indexing or you implement subgraph isomorphism.
Finally, classify:

```python 
classify.py \
    --ftrain output/mutag_features_train.npy --ftest  output/mutag_features_test.npy --ltrain q3_datasets/Mutagenicity/train_labels.txt --ltest  q3_datasets/Mutagenicity/test_labels.txt --proba  output/mutag_probs.npy
```
    --ftrain output/mutag_features_train.npy:  2D feature matrix for the training set.
    --ftest output/mutag_features_test.npy:  2D feature matrix for the test set.
    --ltrain q3_datasets/Mutagenicity/train_labels.txt: Labels for the training set (one per line).
    --ltest q3_datasets/Mutagenicity/test_labels.txt: Labels for the test set (one per line).
    --proba output/mutag_probs.npy: Where to save the predicted probabilities of class 1 on the test set.