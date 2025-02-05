#!/usr/bin/env python3

import sys

def preprocess_for_gspan(input_file, output_file):
    """
    Reads graphs from 'input_file' in a format that uses '#'
    to indicate the start of each graph, and lines of the form:
        v <node_id> <node_label>
        e <src> <dst> <edge_label>
    (possibly repeated edges in both directions for an undirected graph).
    Writes an output file for gSpan with lines:
        t # <graph_id>
        v <node_id> <node_label>
        e <src> <dst> <edge_label>
    ensuring no duplicate edges for an undirected graph, 
    and numbering graphs from 0,1,2,...
    """

    in_graph = False
    graph_id = 0
    edges = set()  # store (min_src, max_dst, label) to avoid duplicates
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

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    preprocess_for_gspan(input_file, output_file)
    print(f"Preprocessing complete. Saved gSpan-format file to: {output_file}")

if __name__ == "__main__":
    main()
