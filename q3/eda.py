import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Draw

def load_graph_data(graph_path, label_path):
    """Parse TUDataset format files"""
    graphs = []
    current_graph = None
    
    with open(graph_path) as f:
        for line in f:
            if line.startswith('#'):
                if current_graph is not None:
                    graphs.append(current_graph)
                current_graph = nx.Graph()
            elif line.startswith('v'):
                _, node_id, label = line.strip().split()
                current_graph.add_node(int(node_id), label=int(label))
            elif line.startswith('e'):
                _, u, v, label = line.strip().split()
                current_graph.add_edge(int(u), int(v), label=int(label))
        graphs.append(current_graph)  # Add last graph
    
    labels = np.loadtxt(label_path)
    return graphs, labels

def basic_graph_stats(graphs):
    """Calculate fundamental graph statistics"""
    stats = {
        'num_nodes': [],
        'num_edges': [],
        'node_labels': defaultdict(int),
        'edge_labels': defaultdict(int),
        'degrees': [],
        'connected_components': []
    }
    
    for g in graphs:
        stats['num_nodes'].append(g.number_of_nodes())
        stats['num_edges'].append(g.number_of_edges())
        stats['connected_components'].append(nx.number_connected_components(g))
        
        # Node labels
        for _, data in g.nodes(data=True):
            stats['node_labels'][data['label']] += 1
            
        # Edge labels and degrees
        for u, v, data in g.edges(data=True):
            stats['edge_labels'][data['label']] += 1
            stats['degrees'].append(g.degree(u))
            stats['degrees'].append(g.degree(v))
            
    return stats

def visualize_molecule(graph):
    """Convert graph to RDKit molecule (for chemical datasets)"""
    mol = Chem.RWMol()
    node_map = {}
    
    # Add atoms
    for node, data in graph.nodes(data=True):
        atom = Chem.Atom(int(data['label']))  # Assuming labels map to atomic numbers
        node_map[node] = mol.AddAtom(atom)
    
    # Add bonds
    for u, v, data in graph.edges(data=True):
        bond_type = Chem.BondType.values[int(data['label'])]  # Map to bond types
        mol.AddBond(node_map[u], node_map[v], bond_type)
        
    return mol

# Load both training datasets
train1_graphs, train1_labels = load_graph_data("q3_datasets/Mutagenicity/graphs.txt", "q3_datasets/Mutagenicity/labels.txt")
train2_graphs, train2_labels = load_graph_data("q3_datasets/NCI-H23/graphs.txt", "q3_datasets/NCI-H23/labels.txt")

# 1. Basic Statistics Comparison
stats1 = basic_graph_stats(train1_graphs)
stats2 = basic_graph_stats(train2_graphs)

# Plot distributions
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Node count distribution
sns.histplot(stats1['num_nodes'], ax=ax[0,0], color='blue', label='Dataset 1', kde=True)
sns.histplot(stats2['num_nodes'], ax=ax[0,0], color='orange', label='Dataset 2', kde=True)
ax[0,0].set_title('Node Count Distribution')

# Edge count distribution
sns.histplot(stats1['num_edges'], ax=ax[0,1], color='blue', label='Dataset 1', kde=True)
sns.histplot(stats2['num_edges'], ax=ax[0,1], color='orange', label='Dataset 2', kde=True)
ax[0,1].set_title('Edge Count Distribution')

# Class distribution
sns.countplot(x=np.concatenate([train1_labels, train2_labels]), 
             ax=ax[1,0], hue=np.array(['D1']*len(train1_labels) + ['D2']*len(train2_labels)))
ax[1,0].set_title('Class Distribution Across Datasets')

# Edge label distribution
edge_labels = list(stats1['edge_labels'].keys()) + list(stats2['edge_labels'].keys())
counts = list(stats1['edge_labels'].values()) + list(stats2['edge_labels'].values())
sns.barplot(x=edge_labels, y=counts, ax=ax[1,1])
ax[1,1].set_title('Edge Label Distribution')

plt.tight_layout()
plt.show()

# 2. Molecular Visualization (for chemical datasets)
sample_graphs = train1_graphs[:5] + train2_graphs[:5]
mols = [visualize_molecule(g) for g in sample_graphs]
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,200))
img.show()

# 3. Advanced Structural Analysis
def structural_analysis(graphs):
    analysis = {
        'cycle_basis': [],
        'clustering_coeff': [],
        'degree_centrality': []
    }
    
    for g in graphs:
        # Convert to undirected for analysis
        ug = g.to_undirected()
        
        # Cycle analysis
        analysis['cycle_basis'].append(len(nx.cycle_basis(ug)))
        
        # Clustering coefficient
        analysis['clustering_coeff'].append(nx.average_clustering(ug))
        
        # Degree centrality
        analysis['degree_centrality'].append(np.mean(list(nx.degree_centrality(ug).values())))
    
    return analysis

struct1 = structural_analysis(train1_graphs)
struct2 = structural_analysis(train2_graphs)

# Plot structural features
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=[struct1['cycle_basis'], struct2['cycle_basis']], 
            ax=ax[0])
ax[0].set_title('Cycle Basis Sizes')

sns.violinplot(data=[struct1['clustering_coeff'], struct2['clustering_coeff']],
              ax=ax[1])
ax[1].set_title('Clustering Coefficients')

sns.kdeplot(struct1['degree_centrality'], ax=ax[2], label='Dataset 1')
sns.kdeplot(struct2['degree_centrality'], ax=ax[2], label='Dataset 2')
ax[2].set_title('Degree Centrality Distribution')

plt.tight_layout()
plt.show()
