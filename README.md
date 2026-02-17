# Data-Miners

Course assignments from IIT Delhi covering data mining and AI topics.

## Repository Structure

### hw1 -- Frequent Pattern Mining (Data Mining)

Benchmarking and applying frequent subgraph/itemset mining algorithms.

| Folder | Description |
|--------|-------------|
| `q1/`  | Apriori vs FP-Growth comparison -- benchmarks runtime across support thresholds and generates timing plots |
| `q2/`  | Frequent subgraph mining -- compares gSpan, FSG, and Gaston on molecular graph datasets with timing analysis |
| `q3/`  | Discriminative subgraph-based graph classification -- mines frequent subgraphs with gSpan, ranks them by information gain, filters redundancy via Jaccard similarity, builds a presence-matrix feature representation, and trains an SVM classifier (ROC-AUC evaluation) |

**Usage (q1 example):**
```bash
cd hw1/q1
bash q1.sh <apriori_exe> <fp_exe> <dataset> <output_dir>
```

### hw2 -- Influence Maximization (COL761 Data Mining)

Greedy seed selection for the Independent Cascade (IC) model on social networks.

- `src/solution.cpp` -- OpenMP-parallelized greedy algorithm that pre-generates Monte Carlo graph instances, then iteratively picks the node with the highest marginal spread gain.
- `report.pdf` -- NP-hardness proof (reduction from Set Cover), algorithm design and complexity analysis, and a counterexample showing sub-optimality of the greedy approach.

**Build & run:**
```bash
cd hw2
g++ -O2 -fopenmp -o solution src/solution.cpp
./solution <graph_file> <output_file> <k> <num_simulations>
```

### hw3 -- Node Classification with Graph Neural Networks (Data Mining)

Two GNN-based prediction tasks on graph-structured data.

**Task 1 -- Node classification on homogeneous graphs** (`model1_d1.py`, `model1_d2.py`)
- GIN + GraphSAGE hybrid architecture with JumpingKnowledge aggregation, residual connections, and batch normalization.
- Includes cosine-annealed LR with warmup and gradient clipping.
- `prepare_task1.py` converts raw numpy arrays (edges, features, labels) into a PyTorch Geometric `Data` object.

**Task 2 -- User personality prediction on bipartite graphs** (`model2.py`)
- Bipartite user-product interaction graph with separate feature projections for user and product nodes.
- Two-layer GraphSAGE with batch normalization, dropout, and residual skip connections.
- Trained with BCEWithLogitsLoss for multi-label classification.
- `prepare_task2.py` cleans data by filtering NaN-labeled users and re-indexing the bipartite edge list.

**Setup:**
```bash
cd hw3/setupfiles
conda env create -f environment.yml
conda activate myenv
```

**Train & test (example):**
```bash
bash train1_d1.sh   # train task-1 model on dataset 1
bash test1_d1.sh    # evaluate
bash train2.sh      # train task-2 bipartite GNN
bash test2.sh       # evaluate
```

## Tech Stack

- Python, NumPy, scikit-learn, NetworkX
- PyTorch, PyTorch Geometric (GraphSAGE, GIN, JumpingKnowledge)
- C++ with OpenMP (influence maximization)
- External tools: Apriori, FP-Growth, gSpan, FSG, Gaston

## Authors

Arnav Raj, Ayush Gupta, Arpit Agrawal -- IIT Delhi
