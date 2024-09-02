# import numpy as np
# import torch

# def check_whole_graph_txt(file_path):
#     print(f"Checking {file_path}...")
#     try:
#         edges = np.loadtxt(file_path, dtype=np.int64, delimiter=',')
#         print(f"Loaded {edges.shape[0]} edges from {file_path}")
#         print("First few edges:", edges[:5])
        
#         # Check for duplicate edges
#         unique_edges = np.unique(edges, axis=0)
#         if edges.shape[0] != unique_edges.shape[0]:
#             print(f"Warning: {file_path} contains duplicate edges")
#         else:
#             print(f"{file_path} does not contain duplicate edges")
        
#         # Check for invalid node indices (assuming nodes are non-negative)
#         if np.any(edges < 0):
#             print(f"Warning: {file_path} contains invalid node indices (negative values)")
#         else:
#             print(f"{file_path} does not contain invalid node indices")
        
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")

# def check_train_di_txt(file_path):
#     print(f"Checking {file_path}...")
#     try:
#         edges = np.loadtxt(file_path, dtype=np.int64, delimiter=',')
#         print(f"Loaded {edges.shape[0]} edges from {file_path}")
#         print("First few edges:", edges[:5])
        
#         # Check for duplicate edges
#         unique_edges = np.unique(edges, axis=0)
#         if edges.shape[0] != unique_edges.shape[0]:
#             print(f"Warning: {file_path} contains duplicate edges")
#         else:
#             print(f"{file_path} does not contain duplicate edges")
        
#         # Check for invalid node indices (assuming nodes are non-negative)
#         if np.any(edges[:, :2] < 0):
#             print(f"Warning: {file_path} contains invalid node indices (negative values)")
#         else:
#             print(f"{file_path} does not contain invalid node indices")
        
#         # Check for invalid labels (assuming labels are 0, 1, 2, or 3)
#         valid_labels = [0, 1, 2, 3]
#         if not np.all(np.isin(edges[:, 2], valid_labels)):
#             print(f"Warning: {file_path} contains invalid labels")
#         else:
#             print(f"{file_path} does not contain invalid labels")
        
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")

# Example usage
dataset = 'human'
seed = 0
task = 1

whole_graph_file = f'./edge_data/{dataset}/whole.graph.txt'
train_di_file = f'./edge_data/{dataset}/{seed}/train_di.txt'

# check_whole_graph_txt(whole_graph_file)
# check_train_di_txt(train_di_file)


import numpy as np

def load_edges_from_file1(file_path):
    try:
        edges = np.loadtxt(file_path, dtype=int, delimiter=' ')
        if edges.ndim == 1:  # If only one edge present, reshape to 2D
            edges = edges.reshape(1, -1)
        return edges
    except Exception as e:
        print(f"Error loading edges from {file_path}: {e}")
        return None
    

def load_edges_from_file2(file_path):
    try:
        edges = np.loadtxt(file_path, dtype=int, delimiter=',')
        if edges.ndim == 1:  # If only one edge present, reshape to 2D
            edges = edges.reshape(1, -1)
        return edges
    except Exception as e:
        print(f"Error loading edges from {file_path}: {e}")
        return None

def check_redundant_edges(graph_edges, filename):
    """
    Check for redundant (duplicate) edges in the graph and classify them by the number of redundancies.
    
    Parameters:
    - graph_edges (np.ndarray): The array of graph edges.
    - filename (str): The filename for reference.
    
    Returns:
    - None: Prints the count of redundant edges and the redundant edges themselves.
    """
    # Convert the array of edges to a list of tuples
    edge_list = [tuple(edge) for edge in graph_edges.tolist()]
    
    # Use a dictionary to count occurrences of each edge
    edge_count = {}
    for edge in edge_list:
        if edge in edge_count:
            edge_count[edge] += 1
        else:
            edge_count[edge] = 1
    
    # Classify redundant edges by their redundancy count
    redundancy_classification = {}
    total_redundancy_count = 0
    for edge, count in edge_count.items():
        if count > 1:
            if count not in redundancy_classification:
                redundancy_classification[count] = []
            redundancy_classification[count].append(edge)
            total_redundancy_count += (count - 1)  # Each additional occurrence of an edge is a redundancy
    
    if len(redundancy_classification) == 0:
        print(f"No redundant edges found in {filename}.")
    else:
        print(f"Redundant edges found in {filename}:")
        for count, edges in sorted(redundancy_classification.items()):
            print(f"Edges with {count - 1} redundancy(ies) (appeared {count} times): {edges}")
        print(f"Total number of redundant edges: {total_redundancy_count}")
        print(f"Total redundant edge instances: {sum(len(edges) for edges in redundancy_classification.values())}")

def check_edges_in_graph(graph_edges, test_edges, filename):
    # Check if all test edges exist in the graph
    missing_edges = []
    for edge in test_edges:
        src, dst = edge[0], edge[1]
        if not ((graph_edges[:, 0] == src) & (graph_edges[:, 1] == dst)).any():
            missing_edges.append((src, dst))
    
    if len(missing_edges) == 0:
        print(f"All edges in {filename} exist in the graph.")
    else:
        print(f"Missing edges in {filename}: {missing_edges}")
        print(f"Total missing edges: {len(missing_edges)}")

def count_unique_nodes(edges):
    """
    Count the number of unique nodes in the edges array.
    
    Parameters:
    - edges (np.ndarray): The array of graph edges.
    
    Returns:
    - int: The number of unique nodes.
    """
    unique_nodes = np.unique(edges.flatten())
    return len(unique_nodes)

def main():
    # Load edges from whole.graph.txt
    graph_edges = load_edges_from_file1(whole_graph_file)
    if graph_edges is None:
        return

    # Check for redundant edges in whole.graph.txt
    check_redundant_edges(graph_edges, 'whole.graph.txt')

    # Count and print the number of unique nodes
    num_nodes = count_unique_nodes(graph_edges)
    print(f"Total number of unique nodes in whole.graph.txt: {num_nodes}")

    # Load edges from train_di.txt (or any other file you want to check)
    train_di_edges = load_edges_from_file2(train_di_file)
    if train_di_edges is None:
        return
    
    # Count and print the number of unique nodes in train_di.txt
    train_di_num_nodes = count_unique_nodes(train_di_edges)
    print(f"Total number of unique nodes in train_di.txt: {train_di_num_nodes}")
    
    # Check if train_di_edges exist in graph_edges
    check_edges_in_graph(graph_edges, train_di_edges, 'train_di.txt')

if __name__ == "__main__":
    main()
