#!/usr/bin/env python3
import argparse
import subprocess
import networkx as nx
import numpy as np
import os
from math import floor
from scipy.stats import chi2_contingency

##############################################################################
# 1. Loading Graphs in gSpan Format
##############################################################################
def load_graphs_gspan_format(path_file):
    """
    Expects lines like:
      t # 0
      v 0 <label>
      v 1 <label>
      e 0 1 <label>
      t # 1
      ...
    Returns list of networkx.Graph, each with .graph['id'] = ID.
    """
    graphs = []
    G = None

    with open(path_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('t #'):
                # if we already had a graph, save it
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()
                # parse the ID
                parts = line.split('#')
                graph_id = parts[1].strip().split()[0]
                G.graph['id'] = int(graph_id)

            else:
                parts = line.split()
                if parts[0] == 'v':
                    node_id = int(parts[1])
                    node_label = parts[2]
                    G.add_node(node_id, label=node_label)
                elif parts[0] == 'e':
                    s = int(parts[1])
                    d = int(parts[2])
                    elabel = parts[3]
                    G.add_edge(s, d, label=elabel)

    # last graph
    if G is not None:
        graphs.append(G)

    return graphs

def load_labels(path_labels):
    """
    Reads one integer label per line (0 or 1).
    Returns a numpy array of shape (num_graphs,).
    """
    with open(path_labels, 'r') as f:
        labs = [int(line.strip()) for line in f if line.strip()]
    return np.array(labs, dtype=int)

##############################################################################
# 2. Run gSpan
##############################################################################
def run_gspan(path_gspan_binary, path_input, min_support):
    """
    Calls gSpan:  gSpan -f <path_input> -s <(min_support *100)> -o -i
    => gSpan writes <path_input>.fp by default.
    Returns that .fp filepath.
    """
    out_fp = path_input + ".fp"
    if os.path.exists(out_fp):
        os.remove(out_fp)

    pct = min_support * 100.0
    cmd = [
        path_gspan_binary,
        '-f', path_input,
        '-s', str(pct),
        '-o',  # write discovered subgraphs to <path_input>.fp
        '-i'   # lines with "x:" indicating which graph IDs contain pattern
    ]
    print("[INFO] Running gSpan with command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[INFO] gSpan completed. Output is {out_fp}")
    return out_fp

##############################################################################
# 3. Parse .fp file for subgraphs
##############################################################################
def parse_fp_file(path_fp, ngraphs):
    """
    We parse lines like:
      t # <patternID> * <supportCount>
      ...
      x: <list_of_graphIDs>
    Store each subgraph as a dict: {'id':..., 'support':..., 'graphs': set(...)}.
    """
    subgraphs = []
    current_subg = None

    with open(path_fp, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith('t #'):
                # store old subg if any
                if current_subg is not None:
                    subgraphs.append(current_subg)

                # e.g. "t # 0 * 64110"
                # or "t # 1"
                parts = line.split()
                pattern_id = parts[2]  # e.g. '0'
                support_count = None
                if len(parts) >= 4 and parts[3] == '*':
                    if len(parts) >= 5:
                        # e.g. "* 64110"
                        support_count = int(parts[4])
                current_subg = {
                    'id': pattern_id,
                    'support': support_count,
                    'graphs': set()
                }

            elif line.startswith('x:'):
                # e.g. "x: 0 2 5"
                tokens = line[2:].strip().split()
                for tok in tokens:
                    if tok.isdigit():
                        current_subg['graphs'].add(int(tok))
            # else lines with v or e describing subgraph structure => skip or parse if needed

    if current_subg is not None:
        subgraphs.append(current_subg)

    return subgraphs

##############################################################################
# 4. Chi-square
##############################################################################
def compute_chi2(subg, labels, ngraphs):
    present_label0 = 0
    present_label1 = 0
    absent_label0  = 0
    absent_label1  = 0

    for i in range(ngraphs):
        is_present = (i in subg['graphs'])
        lbl = labels[i]
        if is_present:
            if lbl == 0:
                present_label0 += 1
            else:
                present_label1 += 1
        else:
            if lbl == 0:
                absent_label0 += 1
            else:
                absent_label1 += 1

    table = np.array([
        [present_label0, present_label1],
        [absent_label0, absent_label1]
    ], dtype=int)
    # Edge case: If expected freq is 0 => ValueError => we can do a small fix:
    # e.g., add 1 to each cell or catch the error
    try:
        chi2, pval, _, _ = chi2_contingency(table)
    except ValueError:
        # If the table has a zero in an expected freq => skip or set chi2=0
        chi2 = 0.0
    return chi2

def rank_subgraphs_by_chi2(subgraphs, labels, ngraphs):
    scored = []
    for sg in subgraphs:
        val = compute_chi2(sg, labels, ngraphs)
        scored.append((sg, val))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored]

##############################################################################
# 5. Convert => build presence/absence
##############################################################################
def build_presence_matrix(subgraphs, ngraphs):
    """
    Build an array [n, len(subgraphs)] with presence=1 if i in subg['graphs'].
    This only works if the graph indices used in .fp match the new dataset.
    """
    feat = np.zeros((ngraphs, len(subgraphs)), dtype=np.uint8)
    for j, sg in enumerate(subgraphs):
        for gidx in sg['graphs']:
            if 0 <= gidx < ngraphs:
                feat[gidx, j] = 1
    return feat

##############################################################################
# 6. Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["identify","convert"], required=True)
    parser.add_argument("--graphs", required=True,
                        help="Path to the gSpan-format input file for identify/convert.")
    parser.add_argument("--labels", default=None,
                        help="Path to the label file (only needed in identify).")
    parser.add_argument("--binary", default=None,
                        help="Path to gSpan executable (only needed in identify).")
    parser.add_argument("--min_support", type=float, default=0.05,
                        help="Minimum support fraction (e.g. 0.05 => 5%).")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Keep top K subgraphs in identify mode.")
    parser.add_argument("--out_features", default=None,
                        help="Where to save the presence/absence .npy in convert mode.")
    args = parser.parse_args()

    if args.mode == "identify":
        if not args.labels or not args.binary:
            raise ValueError("In 'identify' mode, must provide --labels and --binary.")
        # 1) Load graphs + labels
        Gs = load_graphs_gspan_format(args.graphs)
        labels = load_labels(args.labels)
        n = len(Gs)
        print(f"[INFO] Loaded {n} graphs. labels shape={labels.shape}")

        # 2) Run gSpan => writes <args.graphs>.fp
        fp_path = run_gspan(args.binary, args.graphs, args.min_support)

        # 3) Parse .fp
        subgraphs = parse_fp_file(fp_path, n)
        print(f"[INFO] Found {len(subgraphs)} subgraphs in {fp_path}.")

        # 4) rank by chi2
        ranked = rank_subgraphs_by_chi2(subgraphs, labels, n)
        top_subs = ranked[:args.top_k]
        print(f"[INFO] top {len(top_subs)} subgraphs by chi2 => printing info:")
        for i, sg in enumerate(top_subs):
            print(f" Subg #{i}, ID={sg['id']}, support={sg['support']}, #graphs={len(sg['graphs'])}")

        print("[INFO] identify mode complete.")

    elif args.mode == "convert":
        if not args.out_features:
            raise ValueError("In 'convert' mode, must specify --out_features for the .npy matrix.")

        # 1) We load the same gSpan-format graphs
        Gs = load_graphs_gspan_format(args.graphs)
        n = len(Gs)
        print(f"[INFO] Loaded {n} graphs in convert mode from {args.graphs}")

        # 2) Parse the .fp from the same prefix
        fp_path = args.graphs + ".fp"
        if not os.path.exists(fp_path):
            print(f"[WARN] {fp_path} does not exist => returning zero features.")
            feats = np.zeros((n, 1), dtype=np.uint8)
        else:
            subgraphs = parse_fp_file(fp_path, n)
            feats = build_presence_matrix(subgraphs, n)
            print(f"[INFO] Built feature matrix with shape={feats.shape}")

        # 3) Save
        np.save(args.out_features, feats)
        print(f"[INFO] Saved features to {args.out_features} shape={feats.shape}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
