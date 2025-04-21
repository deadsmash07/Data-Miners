# HW3: Readme
```plaintext
.
└── hw3
    ├── eval_task1_d1.py
    ├── hw3_col761_2025_datasets
    │   └── datasets
    │       ├── task1
    │       │   ├── d1
    │       │   │   ├── edges.npy
    │       │   │   ├── label.npy
    │       │   │   └── node_feat.npy
    │       │   ├── d2
    │       │   │   ├── edges.npy
    │       │   │   ├── label.npy
    │       │   │   └── node_feat.npy
    │       │   ├── Makefile
    │       │   ├── README.md
    │       │   └── test_shapes.py
    │       └── task2
    │           ├── Makefile
    │           ├── README.md
    │           ├── test_shapes.py
    │           └── train
    │               ├── label.npy
    │               ├── product_features.npy
    │               ├── user_features.npy
    │               └── user_product.npy
    ├── hw3.pdf
    ├── model1_d1.pt
    ├── preds1_d1.csv
    ├── prepare_Data.py
    ├── readme.md
    ├── src
    │   ├── model1.py
    │   ├── model2.py
    │   ├── test1.py
    │   ├── test2.py
    │   ├── train1.py
    │   └── train2.py
    ├── test1.sh
    ├── test2.sh
    ├── train1_d1.pt
    ├── train1.sh
    └── train2.sh
```

## Environment Setup

```bash
conda create -n hw3 python=3.9 -y
conda activate hw3
pip install torch==2.2.1 torch-geometric==2.3.1 scikit-learn pandas numpy
```

---

## Task 1: Node Classification

### Prepare d1

```bash
python3 scripts/prepare_task1.py \
    --edges    ../hw3_col761_2025_datasets/datasets/task1/d1/edges.npy \
    --features ../hw3_col761_2025_datasets/datasets/task1/d1/node_feat.npy \
    --labels   ../hw3_col761_2025_datasets/datasets/task1/d1/label.npy \
    --output   task1_d1.pt
```

### Prepare d2

```bash
python3 scripts/prepare_task1.py \
    --edges    ../hw3_col761_2025_datasets/datasets/task1/d2/edges.npy \
    --features ../hw3_col761_2025_datasets/datasets/task1/d2/node_feat.npy \
    --labels   ../hw3_col761_2025_datasets/datasets/task1/d2/label.npy \
    --output   task1_d2.pt
```

---

## Task 2: Data Preparation

```bash
python3 scripts/prepare_task2.py \
    --user-feats ../hw3_col761_2025_datasets/datasets/task2/train/user_features.npy \
    --prod-feats ../hw3_col761_2025_datasets/datasets/task2/train/product_features.npy \
    --edges      ../hw3_col761_2025_datasets/datasets/task2/train/user_product.npy \
    --labels     ../hw3_col761_2025_datasets/datasets/task2/train/label.npy \
    --output     task2_train.pt
```

---

## Training

### Task 1

```bash
bash train1.sh task1_d1.pt model1_d1.pt
bash train1.sh task1_d2.pt model1_d2.pt
```

### Task 2

```bash
bash train2.sh task2_train.pt model2.pt
```

---

## Testing / Inference

### Task 1

- **d1:** Save probabilities (n×C)

    ```bash
    bash test1.sh task1_d1.pt model1_d1.pt preds1_d1_probs.csv
    ```

- **d2:** Save hard labels (n×1)

    ```bash
    bash test1.sh task1_d2.pt model1_d2.pt preds1_d2.csv
    ```

### Task 2

```bash
bash test2.sh task2_test.pt model2.pt preds2.csv
```

---

## Evaluation

### Task 1 – d1 (ROC‑AUC)

```bash
python3 eval_task1_d1.py
```
### Task 1 – d2 (F1)

```bash
python3 eval_task1_d2.py
```