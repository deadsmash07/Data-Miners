import argparse
import networkx as nx
import numpy as np
import concurrent.futures

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
                    node_label = int(parts[2])
                    G.add_node(node_id, label=node_label)
                elif parts[0] == 'e':
                    s = int(parts[1])
                    d = int(parts[2])
                    elabel = int(parts[3])  
                    G.add_edge(s, d, label=elabel)

    # last graph
    if G is not None:
        graphs.append(G)

    return graphs

def parse_filtered_graphs(file_path):
    """
    We parse lines like:
      t # <patternID> * <supportCount>
      ...
      x: <list_of_graphIDs>
    Store each subgraph as a dict: {'id':..., 'support':..., 'graphs': set(...)}.
    """
    subgraphs = []
    current_graph = None

    with open(file_path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Start of a new subgraph
            if line.startswith('t #'):
                # Store the previous graph before creating a new one
                if current_graph is not None:
                    subgraphs.append(current_graph)

                parts = line.split()
                pattern_id = int(parts[2])  # Extract graph ID
                
                # Create a new NetworkX Graph
                current_graph = nx.Graph()
                current_graph.graph['id'] = pattern_id  # Store ID in graph metadata

            elif line.startswith('v'):
                parts = line.split()
                node_id = int(parts[1])
                node_label = int(parts[2])
                current_graph.add_node(node_id, label=node_label)  # Add node with label

            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edge_label = int(parts[3])
                current_graph.add_edge(v1, v2, label=edge_label)  # Add edge with label

    # Add the last graph
    if current_graph is not None:
        subgraphs.append(current_graph)

    return subgraphs

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

def node_label_match(n1, n2):
    """Return True if node labels match."""
    return n1['label'] == n2['label']

def edge_label_match(e1, e2):
    """Return True if edge labels match."""
    return e1['label'] == e2['label']

def process_graph(args):
    """
    Processes a single graph to check subgraph isomorphism against each filtered subgraph.
    :param args: Tuple (i, graph, filtered_subgraphs)
    :return: Tuple (i, row) where 'row' is a 1D NumPy array indicating for each filtered subgraph
            whether it is present (1) or not (0) in the graph.
    """

    i, G, filtered_subgraphs = args
    row = np.zeros(len(filtered_subgraphs), dtype=np.uint8)
    
    for j, subgraph in enumerate(filtered_subgraphs):
        GM = nx.algorithms.isomorphism.GraphMatcher(
            G, subgraph,
            node_match=node_label_match,
            edge_match=edge_label_match
        )

        if GM.subgraph_is_isomorphic():
            row[j] = 1

    return i, row

def build_presence_matrix(graphs, filtered_subgraphs):
    """
    Builds a 2D NumPy array where each row represents a graph and each column represents a filtered subgraph.
    Value is 1 if the subgraph is present in the graph, otherwise 0.
    
    :param graphs: List of NetworkX graphs (each with node and edge labels).
    :param filtered_subgraphs: List of filtered subgraphs (each with node and edge labels).
    :param n_graphs: Total number of graphs (length of the 'graphs' list).
    :return: 2D NumPy array of shape (n_graphs, len(filtered_subgraphs)) indicating presence (1) or absence (0).
    """
    print("Starting conversion to presence matrix")
    n_graphs = len(graphs)
    presence_matrix = np.zeros((n_graphs, len(filtered_subgraphs)), dtype=np.uint8)
    
    # Create an iterator of arguments for each graph
    args_iter = ((i, G, filtered_subgraphs) for i, G in enumerate(graphs))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for i, row in executor.map(process_graph, args_iter):
            presence_matrix[i, :] = row
    
    return presence_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", required=True,
                        help="Path to the gSpan-format input file for identify/convert.")
    parser.add_argument("--path_discriminative_subgraphs", default=None,
                        help="Path to filtered subgraphs file.")
    parser.add_argument("--out_features", default=None, 
                        help="Path to store output .npy file")
    args = parser.parse_args()

    # Preprocess the input graphs file to the proper format
    preprocessed_graph_path = './preprocess_graph_convert'
    preprocess_for_gspan(args.graphs, preprocessed_graph_path)

    # Load graphs and filtered subgraphs
    Gs = load_graphs_gspan_format(preprocessed_graph_path)
    filtered_subgraphs = parse_filtered_graphs(args.path_discriminative_subgraphs)
    
    # Build the presence matrix in parallel
    numpy_array = build_presence_matrix(Gs, filtered_subgraphs)

    # Save the resulting NumPy array to disk
    np.save(args.out_features, numpy_array)