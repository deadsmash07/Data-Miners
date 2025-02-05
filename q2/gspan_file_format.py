import sys 
import subprocess
import matplotlib.pyplot as plt 
import time

path_dataset = sys.argv[1]
path_gspan_dataset = sys.argv[2]

def convert_graph_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        index = 0
        while index < len(lines):
            if lines[index].startswith('#'):
                graph_id = lines[index].strip()[1:]  # Extracting graph ID
                index += 1
                num_nodes = int(lines[index].strip())  # Number of nodes
                index += 1
                
                # Read node labels
                node_labels = []
                for _ in range(num_nodes):
                    node_labels.append(lines[index].strip())
                    index += 1

                num_edges = int(lines[index].strip())  # Number of edges
                index += 1
                edges = []
                
                # Read edges
                for _ in range(num_edges):
                    src, dest, label = map(int, lines[index].strip().split())
                    edges.append((min(src, dest), max(src, dest), label))
                    index += 1

                # Sort edges by (ID1, ID2)
                edges.sort()
                
                # Write to output file
                outfile.write(f't # {graph_id}\n')
                for node_id, label in enumerate(node_labels):
                    outfile.write(f'v {node_id} {label}\n')
                for src, dest, label in edges:
                    outfile.write(f'e {src} {dest} {label}\n')
            else:
                index+=1

# Example usage
convert_graph_format(path_dataset, path_gspan_dataset)
