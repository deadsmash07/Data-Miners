#!/usr/bin/env python3
import argparse
import subprocess
import networkx as nx
import numpy as np
import os
from math import floor
from scipy.stats import chi2_contingency

##############################################################################
# 1. Loading graphs from a file that uses gSpan's "t # i, v <node>, e <edge>"
##############################################################################
def load_graphs(path_graphs):
    """
    Expects each graph in the file to begin with:
      t # <graph_id>
    Then lines like:
      v <node_id> <node_label>
      e <src> <dst> <edge_label>
    This is the standard gSpan input format described in your README.

    Returns a list of networkx.Graph objects, each corresponding to one "t # i" block.
    """
    graphs = []
    G = None

    with open(path_graphs, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # "t # i" => start new graph
            if line.startswith('t #'):
                # If we were already reading a graph, store it
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()
            else:
                parts = line.split()
                if parts[0] == 'v':
                    # v <node_id> <node_label>
                    node_id = int(parts[1])
                    node_label = parts[2]
                    G.add_node(node_id, label=node_label)
                elif parts[0] == 'e':
                    # e <src> <dst> <edge_label>
                    src = int(parts[1])
                    dst = int(parts[2])
                    edge_label = parts[3]
                    # Undirected edge
                    G.add_edge(src, dst, label=edge_label)

    # If there's a graph in progress, add it
    if G is not None:
        graphs.append(G)

    return graphs

def load_labels(path_labels):
    """
    Reads one label (0 or 1) per line. Returns a numpy array.
    """
    with open(path_labels, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return np.array(labels, dtype=int)

##############################################################################
# 2. Running gSpan externally
##############################################################################
def run_gspan(gspan_binary, dataset_file, output_file, min_support):
    """
    Calls the gSpan binary with:
      gSpan -f <dataset_file> -s <support%> -o -i
    and captures the output in 'output_file'.
    """
    support_percent = min_support * 100.0  # e.g. 0.05 => 5.0

    cmd = [
        gspan_binary,
        '-f', dataset_file,
        '-s', str(support_percent),
        '-o',  # output frequent subgraphs
        '-i'   # include 'x:' lines with graph IDs containing the pattern
    ]
    print("[INFO] Running gSpan:", " ".join(cmd))
    # Run gSpan (output will be in dataset_file.fp)
    subprocess.run(cmd, stderr=subprocess.STDOUT, check=True)
    
    # Construct expected gSpan output file
    gspan_output_file = dataset_file + ".fp"
    
    # Verify the file exists before proceeding
    if not os.path.exists(gspan_output_file):
        raise FileNotFoundError(f"[ERROR] gSpan did not generate expected output file: {gspan_output_file}")

    print(f"[INFO] gSpan completed, results stored in {gspan_output_file}")
    return gspan_output_file  # Return the actual file name

    


##############################################################################
# 3. Parse the gSpan output to find subgraph memberships
##############################################################################
def parse_gspan_output(output_file, ngraphs):
    """
    Reads lines like:
      t # <subgraph_id> * <support_info>
      ...
      x: 0 2 5
    and constructs subgraphs = [{ 'id':..., 'graphs':set([...]) }, ...].
    """
    subgraphs = []
    current_subg = None

    with open(output_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith('t #'):
                # store the old subgraph if exists
                if current_subg is not None:
                    subgraphs.append(current_subg)

                # parse the subgraph id from "t # 0 *..."
                parts = line.split('#')
                sid_part = parts[1].strip()
                # e.g. "0 * 64110"
                subg_id = sid_part.split()[0]
                current_subg = {
                    'id': subg_id,
                    'graphs': set()
                }

            elif line.startswith('x:'):
                # e.g. "x: 0 2 5" => graph IDs
                content = line[2:].strip()
                tokens = content.split()
                for tok in tokens:
                    # skip non-numeric
                    if tok.isdigit():
                        current_subg['graphs'].add(int(tok))

    # add the last subgraph
    if current_subg is not None:
        subgraphs.append(current_subg)

    return subgraphs

##############################################################################
# 4. Score subgraphs by chi-square
##############################################################################
def compute_chi2(subgraphs, labels, ngraphs):
    """
    subgraphs: list of {'id':..., 'graphs':set(...)}.
    labels: array of shape (ngraphs,) with 0/1.
    Returns the subgraphs sorted by descending chi2.
    """
    scored = []
    for sg in subgraphs:
        present_label0 = 0
        present_label1 = 0
        absent_label0 = 0
        absent_label1 = 0
        for i in range(ngraphs):
            is_present = (i in sg['graphs'])
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

        contingency = np.array([
            [present_label0, present_label1],
            [absent_label0, absent_label1]
        ], dtype=np.int32)

        chi2, pval, _, _ = chi2_contingency(contingency)
        scored.append((sg, chi2))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored]

##############################################################################
# 5. Convert any set of graphs to presence/absence
##############################################################################
def convert_to_features(graphs, subgraphs):
    """
    For each subgraph j, presence=1 if the graph i is in subgraph['graphs'].
    Returns a 2D numpy array of shape [num_graphs, num_subgraphs].
    """
    ngraphs = len(graphs)
    nsubs = len(subgraphs)
    features = np.zeros((ngraphs, nsubs), dtype=np.uint8)

    for j, sg in enumerate(subgraphs):
        for gidx in sg['graphs']:
            if 0 <= gidx < ngraphs:
                features[gidx, j] = 1

    return features

##############################################################################
# 6. Main: two modes => identify or convert
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["identify", "convert"], required=True,
                        help="Identify => run gSpan & rank subgraphs; convert => build features.")
    parser.add_argument("--graphs", required=True,
                        help="Path to the gSpan-compatible input file (preprocess_gspan.dat).")
    parser.add_argument("--labels", default=None,
                        help="Path to labels (only used in identify mode).")
    parser.add_argument("--out_subs", default=None,
                        help="Where to store subgraphs after identify.")
    parser.add_argument("--in_subs", default=None,
                        help="Where to read subgraphs in convert mode.")
    parser.add_argument("--out_features", default=None,
                        help="Where to save .npy feature matrix in convert mode.")

    # For identify
    parser.add_argument("--binary", default=None,
                        help="Path to gSpan binary.")
    parser.add_argument("--min_support", type=float, default=0.05,
                        help="Minimum support fraction (0.05 => 5%).")

    args = parser.parse_args()

    if args.mode == "identify":
        # Must have labels, out_subs, and the gSpan binary
        if not args.labels or not args.out_subs or not args.binary:
            raise ValueError("In identify mode, need --labels, --out_subs, and --binary.")

        # 1) Load training graphs + labels from your gSpan format
        graphs = load_graphs(args.graphs)
        labels = load_labels(args.labels)
        ngraphs = len(graphs)

        # 2) Run gSpan
        miner_output = "temp_miner_output.txt"
        run_gspan(
            gspan_binary=args.binary,
            dataset_file=args.graphs,   # directly use the user-provided file
            output_file=miner_output,
            min_support=args.min_support
        )

        # 3) Parse discovered subgraphs
        subgraphs = parse_gspan_output(miner_output, ngraphs)

        # 4) Rank by chi2
        sorted_subs = compute_chi2(subgraphs, labels, ngraphs)

        # 5) Keep top-100
        top_k = 100 if len(sorted_subs) > 100 else len(sorted_subs)
        top_subs = sorted_subs[:top_k]

        # 6) Save them
        with open(args.out_subs, 'w') as fout:
            for sg in top_subs:
                sg_id = sg['id']
                membership = sorted(list(sg['graphs']))
                fout.write(f"SUBG_ID:{sg_id}\n")
                fout.write(f"MEMBERS:{' '.join(map(str, membership))}\n")
                fout.write("----\n")

        print(f"[INFO] Identify done. Wrote {len(top_subs)} subgraphs to {args.out_subs}")

    elif args.mode == "convert":
        # Must have in_subs and out_features
        if not args.in_subs or not args.out_features:
            raise ValueError("In convert mode, need --in_subs and --out_features.")

        # 1) Load the graphs from the gSpan format
        graphs = load_graphs(args.graphs)
        ngraphs = len(graphs)

        # 2) Load subgraphs from in_subs
        subgraphs = []
        with open(args.in_subs, 'r') as fin:
            lines = fin.read().strip().split('\n')

        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith("SUBG_ID:"):
                sg_id = line.split(":", 1)[1].strip()
                idx += 1
                # next line: MEMBERS: ...
                members_line = lines[idx].strip()
                mem_str = members_line.split(":", 1)[1].strip()
                mem_ids = set(map(int, mem_str.split()))
                subgraphs.append({
                    'id': sg_id,
                    'graphs': mem_ids
                })
                idx += 1  # skip ----
            else:
                idx += 1

        # 3) Build features
        features = convert_to_features(graphs, subgraphs)
        print(f"[INFO] Feature matrix shape = {features.shape}")

        # 4) Save
        np.save(args.out_features, features)
        print(f"[INFO] Saved features to {args.out_features}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
