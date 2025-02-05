#!/usr/bin/env python3
import argparse
import subprocess
import networkx as nx
import numpy as np
import os
from math import floor
from scipy.stats import chi2_contingency

##############################################################################
# 1. Loading the assignment’s graph format
##############################################################################
def load_graphs(path_graphs):
    """
    Loads a set of undirected graphs from the format:
      # (new graph)
      v 0 <label>
      v 1 <label>
      ...
      e 0 1 <label>
      # (new graph)
      ...
    Returns: a list of networkx.Graph objects (undirected).
    """
    graphs = []
    G = None
    with open(path_graphs, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # start a new graph
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()  # new undirected graph
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
                    G.add_edge(src, dst, label=edge_label)

    # add the last graph if still in memory
    if G is not None:
        graphs.append(G)
    return graphs

def load_labels(path_labels):
    """
    Loads labels from a file: one label per line (0 or 1).
    Returns a 1D numpy array of shape (num_graphs,).
    """
    with open(path_labels, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return np.array(labels, dtype=int)

##############################################################################
# 2. Run external gSpan binary
##############################################################################
def run_subgraph_miner_gspan(miner_binary, dataset_file, output_file, min_support):
    """
    Calls the gSpan binary to mine frequent subgraphs:
      gSpan -f <dataset_file> -s <support_percent> -o ...
    We redirect stdout to output_file.
    """
    # Convert fractional support to a percentage (e.g., 0.05 => 5.0).
    support_percent = min_support * 100.0

    cmd = [
        miner_binary,
        '-f', dataset_file,
        '-s', str(support_percent),
        '-o',  # output discovered subgraphs to stdout
        '-i'   # include the graph IDs that contain each pattern in the output
    ]
    print("[INFO] Running gSpan command:", " ".join(cmd))

    # We redirect stdout to output_file so we can parse it later
    with open(output_file, 'w') as outf:
        subprocess.run(cmd, stdout=outf, check=True)

    print(f"[INFO] gSpan completed. Results in {output_file}")

##############################################################################
# 3. Parse gSpan output to build subgraph membership
##############################################################################
def parse_mined_subgraphs_gspan(output_file, ngraphs):
    """
    Reads the gSpan output from 'output_file'.
    We expect lines like:
      t # <subg_id> * <support_info>
      ...
      x: <list_of_graphIDs> or something similar
    This is version-dependent. We'll parse for patterns:
      "t # 0 *" => new subgraph with ID=0
      "x: 3  -> SUPPORT: #graphs: 7" might list the containing graphs
    We'll store membership in subgraphs[k]['graphs'] = set([...]).
    The user must adapt this to the actual lines gSpan prints.
    """
    subgraphs = []
    current_subg = None

    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Example: line "t # 0 *..."
            if line.startswith('t #'):
                # if we were reading a subgraph, store it
                if current_subg is not None:
                    subgraphs.append(current_subg)

                # parse the new subg ID
                parts = line.split('#')
                subg_id_str = parts[1].strip()
                # sometimes there's additional info after that
                # e.g. "0 * freq..."
                # We'll just store the first token
                subg_id_str = subg_id_str.split()[0]
                current_subg = {
                    'id': subg_id_str,
                    'graphs': set()
                }

            # Example: a line that starts with 'x: ' listing graph IDs?
            # e.g. "x: 0 2 3" to show which graphs the pattern is in
            elif line.startswith('x:'):
                # parse graph IDs
                # line might be: "x: 0 2 5"
                # or it might have extra text about support
                # We'll just strip off 'x:' and split the rest
                content = line[2:].strip()
                # e.g. "0 2 5"
                tokens = content.split()
                # interpret each token as a graph ID
                # might skip tokens that aren't numeric if the line has more data
                for tok in tokens:
                    if tok.isdigit():
                        current_subg['graphs'].add(int(tok))

    # After file ends, if we have a subgraph in progress
    if current_subg is not None:
        subgraphs.append(current_subg)

    return subgraphs

##############################################################################
# 4. Discriminative filtering by chi-square
##############################################################################
def compute_chi2(subgraphs, labels, ngraphs):
    """
    subgraphs: list of dicts with {'id':..., 'graphs': set([...])}
    labels: array of shape (ngraphs,)
    We return subgraphs sorted by descending chi2 score.
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

        # 2x2 contingency table
        table = np.array([[present_label0, present_label1],
                          [absent_label0, absent_label1]], dtype=np.int32)
        # compute chi2
        chi2, pval, _, _ = chi2_contingency(table)
        scored.append((sg, chi2))

    # sort by chi2 descending
    scored.sort(key=lambda x: x[1], reverse=True)
    top_subgraphs = [x[0] for x in scored]
    return top_subgraphs

##############################################################################
# 5. Convert any set of graphs to presence/absence features
##############################################################################
def convert_to_features(graphs, subgraphs):
    """
    If subgraphs already contain the membership info, we just mark presence=1
    if the index i is in subgraph['graphs'].
    Features: shape [n_graphs, n_subgraphs]
    """
    ngraphs = len(graphs)
    nsubs = len(subgraphs)
    features = np.zeros((ngraphs, nsubs), dtype=np.uint8)

    # We assume the graph index used in membership sets is the same as the order
    # in which we loaded them. That implies that the training/test sets are the
    # same or consistent with these indices.
    for j, sg in enumerate(subgraphs):
        for gidx in sg['graphs']:
            if 0 <= gidx < ngraphs:
                features[gidx, j] = 1

    return features

##############################################################################
# 6. Main Entry: Two modes => "identify" or "convert"
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["identify", "convert"], required=True,
                        help="Mode: 'identify' (mine + rank subgraphs) or 'convert' (create features).")
    parser.add_argument("--graphs", type=str, required=True,
                        help="Path to graphs file in assignment format (#, v, e lines).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels (only needed in identify mode).")
    parser.add_argument("--out_subs", type=str, default=None,
                        help="Path to store subgraphs in identify mode.")
    parser.add_argument("--in_subs", type=str, default=None,
                        help="Path to read subgraphs in convert mode.")
    parser.add_argument("--out_features", type=str, default=None,
                        help="Path to store 2D numpy features in convert mode.")

    # Additional arguments for identify
    parser.add_argument("--binary", type=str, default=None,
                        help="Path to gSpan binary (only needed in identify mode).")
    parser.add_argument("--min_support", type=float, default=0.05,
                        help="Minimum support fraction (0.05 => 5%).")

    args = parser.parse_args()

    if args.mode == "identify":
        # Check required arguments
        if not args.labels or not args.out_subs or not args.binary:
            raise ValueError("In 'identify' mode, must provide --labels, --out_subs, and --binary.")

        # 1) Load training graphs & labels
        graphs = load_graphs(args.graphs)
        labels = load_labels(args.labels)
        ngraphs = len(graphs)

        # 2) Preprocess or copy the input to a gSpan-compatible file
        #    We'll assume you have already preprocessed externally, or we do a direct copy:
        miner_infile = "temp_miner_input.dat"
        # If your data is already in the correct gSpan format, just do:
        # cp <args.graphs> -> miner_infile
        # otherwise, implement a transform function. For example:
        # For now, let's just do a direct copy:
        with open(args.graphs, 'r') as fin, open(miner_infile, 'w') as fout:
            fout.write(fin.read())

        # 3) Run gSpan
        miner_outfile = "temp_miner_output.txt"
        run_subgraph_miner_gspan(
            miner_binary=args.binary,
            dataset_file=miner_infile,
            output_file=miner_outfile,
            min_support=args.min_support
        )

        # 4) Parse mined subgraphs
        subgraphs = parse_mined_subgraphs_gspan(miner_outfile, ngraphs)

        # 5) Discriminative filtering via chi2
        sorted_subs = compute_chi2(subgraphs, labels, ngraphs)

        # 6) Keep top 100
        top_k = 100 if len(sorted_subs) > 100 else len(sorted_subs)
        top_subs = sorted_subs[:top_k]

        # 7) Write them to <args.out_subs>
        with open(args.out_subs, 'w') as f:
            for sg in top_subs:
                sg_id = sg['id']
                membership_list = sorted(list(sg['graphs']))
                f.write(f"SUBG_ID:{sg_id}\n")
                f.write(f"MEMBERS:{' '.join(map(str, membership_list))}\n")
                f.write("----\n")

        print(f"[INFO] Completed identify mode. Wrote {top_k} subgraphs to {args.out_subs}")

    elif args.mode == "convert":
        # Check required arguments
        if not args.in_subs or not args.out_features:
            raise ValueError("In 'convert' mode, must provide --in_subs and --out_features.")

        # 1) Load the graphs
        graphs = load_graphs(args.graphs)
        ngraphs = len(graphs)

        # 2) Load subgraphs from <args.in_subs>
        subgraphs = []
        with open(args.in_subs, 'r') as fin:
            lines = fin.read().strip().split('\n')

        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith("SUBG_ID:"):
                sg_id = line.split(":", 1)[1].strip()
                idx += 1
                # next line: "MEMBERS: ..."
                mem_line = lines[idx].strip()
                membership_str = mem_line.split(":", 1)[1].strip()
                membership_ids = set(map(int, membership_str.split()))
                subgraphs.append({
                    'id': sg_id,
                    'graphs': membership_ids
                })
                idx += 1  # skip "----" line
            else:
                idx += 1

        # 3) Convert to features
        features = convert_to_features(graphs, subgraphs)
        print(f"[INFO] Feature matrix shape = {features.shape}")

        # 4) Save as numpy
        np.save(args.out_features, features)
        print(f"[INFO] Saved features to {args.out_features}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")
