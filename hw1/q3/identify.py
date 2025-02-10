import argparse 
import subprocess
import networkx as nx
import numpy as np
import os
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

    
def subgraph_signature(subg):
    """
    Generates a set of tokens representing the subgraph's structural signature
    based on its node and edge labels.
    
    Here we simply add:
      - A token for each node: "v:{node_label}"
      - A token for each edge in canonical order: "e:{min_node}:{max_node}:{edge_label}"
    
    This is a simple signature that you can adjust depending on how fine-grained
    you want your similarity measure to be.
    """
    sig = set()
    # Add node tokens based on node labels
    for node_id, node_label in subg['vertices'].items():
        sig.add(f"v:{node_label}")
    # Add edge tokens (using canonical ordering for nodes)
    for v1, v2, edge_label in subg['edges']:
        a, b = sorted((v1, v2))
        sig.add(f"e:{a}:{b}:{edge_label}")
    return sig

def redundancy_aware_filter(ranked_subgraphs, threshold=0.8, top_k=100):
    """
    Given a ranked list of subgraphs, this function returns a filtered list that
    avoids redundancy. For each subgraph in the ranked list, it computes the Jaccard 
    similarity (index) between its signature and those of subgraphs already selected.
    
    A subgraph is added to the final "selected" list only if its maximum Jaccard 
    similarity with any already selected subgraph is below the threshold.
    
    The process stops once top_k subgraphs are selected or the ranked list is exhausted.
    """
    selected = []
    # Precompute signatures for all subgraphs
    signatures = {}
    for sg in ranked_subgraphs:
        # We assume each subgraph has a unique 'id'
        signatures[sg['id']] = subgraph_signature(sg)
    
    for sg in ranked_subgraphs:
        sig = signatures[sg['id']]
        is_redundant = False
        # Check similarity with all subgraphs in 'selected'
        for sel_sg in selected:
            sel_sig = signatures[sel_sg['id']]
            # Compute Jaccard index = |intersection| / |union|
            intersection = sig.intersection(sel_sig)
            union = sig.union(sel_sig)
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= threshold:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(sg)
            if len(selected) >= top_k:
                break
    return selected


def compute_information_gain(subg, labels, ngraphs):
    """
    Computes the information gain of a subgraph (treated as a binary feature)
    with respect to the labels.
    
    For each graph i in 0 ... ngraphs-1:
      - If i is in subg['graphs'], the subgraph is present.
      - Otherwise it is absent.
    
    Information gain is computed as:
    
        IG = H(Y) - [ P(X=present)*H(Y|present) + P(X=absent)*H(Y|absent) ]
    
    where H(.) is the entropy (base 2).
    """
    # Count label frequencies when the subgraph is present vs absent.
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

    # Helper to compute entropy for two counts.
    def entropy(count0, count1):
        s = count0 + count1
        if s == 0:
            return 0.0
        p0 = count0 / s
        p1 = count1 / s
        ent = 0.0
        if p0 > 0:
            ent -= p0 * np.log2(p0)
        if p1 > 0:
            ent -= p1 * np.log2(p1)
        return ent

    # Overall entropy H(Y)
    total_label0 = present_label0 + absent_label0
    total_label1 = present_label1 + absent_label1
    H_total = entropy(total_label0, total_label1)

    # Entropy conditioned on subgraph presence/absence.
    present_total = present_label0 + present_label1
    absent_total  = absent_label0  + absent_label1

    H_present = entropy(present_label0, present_label1)
    H_absent  = entropy(absent_label0,  absent_label1)

    # Weighted conditional entropy.
    H_cond = 0.0
    if ngraphs > 0:
        H_cond = (present_total / ngraphs) * H_present + (absent_total / ngraphs) * H_absent

    information_gain = H_total - H_cond
    return information_gain

def rank_subgraphs_by_information_gain(subgraphs, labels, ngraphs):
    """
    Computes the information gain for each subgraph and returns the list
    of subgraphs sorted by descending information gain.
    """
    scored = []
    for sg in subgraphs:
        ig = compute_information_gain(sg, labels, ngraphs)
        scored.append((sg, ig))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored]

def preprocess_for_gspan(input_file, output_file):

    in_graph = False
    graph_id = 0
    edges = set()
    output_lines = []
    
    with open(input_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Encountering '#': means a new graph is starting
            if line.startswith('#'):
                # If we were already reading a graph, we need to "end" it
                # by writing the edges we've collected
                if in_graph:
                    # Dump the edges from the set, skipping duplicates
                    for (s, d, lbl) in edges:
                        output_lines.append(f"e {s} {d} {lbl}")
                    edges.clear()
                
                # Start a new graph
                output_lines.append(f"t # {graph_id}")
                graph_id += 1
                in_graph = True
                
            else:
                # If this line describes a node or edge
                parts = line.split()
                if parts[0] == 'v':
                    # v <node_id> <node_label>
                    node_id = parts[1]
                    node_label = parts[2]
                    output_lines.append(f"v {node_id} {node_label}")
                elif parts[0] == 'e':
                    # e <src> <dst> <edge_label>
                    src = int(parts[1])
                    dst = int(parts[2])
                    elabel = parts[3]
                    # store it once (undirected), so always (min, max)
                    s = min(src, dst)
                    d = max(src, dst)
                    edges.add((s, d, elabel))
                else:
                    # Ignore or handle unexpected lines
                    pass

    # End of file: if we ended reading a graph, flush its edges
    if in_graph and edges:
        for (s, d, lbl) in edges:
            output_lines.append(f"e {s} {d} {lbl}")
        edges.clear()

    # Write final output
    with open(output_file, 'w') as fout:
        fout.write("\n".join(output_lines) + "\n")

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
    # print("[INFO] Running gSpan with command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # print(f"[INFO] gSpan completed. Output is {out_fp}")
    return out_fp

def parse_fp_file(path_fp):
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
                    'vertices' : {},
                    'edges' : [],   
                    'graphs': set()
                }

            elif line.startswith('x'):
                # e.g. "x: 0 2 5"
                tokens = line[2:].strip().split()
                for tok in tokens:
                    if tok.isdigit():
                        current_subg['graphs'].add(int(tok))
            # else lines with v or e describing subgraph structure => skip or parse if needed

            elif line.startswith('v'):
                parts = line.split()
                node_id = int(parts[1])
                node_label = int(parts[2])
                current_subg['vertices'][node_id] = node_label
            
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edge_label = int(parts[3])
                current_subg['edges'].append((v1,v2,edge_label))



    if current_subg is not None:
        subgraphs.append(current_subg)

    return subgraphs


parser = argparse.ArgumentParser()
parser.add_argument("--graphs", required=True,
                    help="Path to the gSpan-format input file for identify/convert.")
parser.add_argument("--labels", default=None,
                    help="Path to the label file (only needed in identify).")
parser.add_argument("--path_discriminative_subgraphs", default = None, 
                    help="Path to store filtered Graphs")

args = parser.parse_args()

preprocessed_graph_path = './preprocess_graph'
preprocess_for_gspan(args.graphs,preprocessed_graph_path)

Gs = load_graphs_gspan_format(preprocessed_graph_path)
labels = load_labels(args.labels)


GSPAN_BINARY_PATH = './gspan'
MIN_SUPPORT = 0.003
frequent_subgraphs_path = run_gspan(GSPAN_BINARY_PATH, preprocessed_graph_path, MIN_SUPPORT)
frequent_subgraphs = parse_fp_file(frequent_subgraphs_path)
ranked = rank_subgraphs_by_information_gain(frequent_subgraphs, labels, len(Gs))
top_k = 100
top_subs = ranked[:top_k]
top_subs = redundancy_aware_filter(ranked, threshold=0.8, top_k=100)


with open(args.path_discriminative_subgraphs , 'w') as file : 
    for i,sg in enumerate(top_subs): 
        file.write(f"t # {sg['id']}\n")
        for node_id, node_label in sg['vertices'].items(): 
            file.write(f"v {node_id} {node_label}\n")
        
        for v1,v2,edge_label in sg['edges']:
            file.write(f"e {v1} {v2} {edge_label}\n") 

