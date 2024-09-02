import data_preprocessing as dp
from config import const_args as args
import dgl
import argparse

args = argparse.Namespace(**args)

# for dataset in ['human']:
#     print(dataset)
#     for task in [3,4]:
#         for seed in range(10):
#             graph = dgl.load_graphs('./edge_data/%s/whole.graph'%(dataset))[0][0]
#             save_path = './edge_data/%s/'%(dataset)
#             dp.split_data(args, graph, save_path, seed, task)

# def load_and_deduplicate_graph(filepath):
#     graph = dgl.load_graphs(filepath)[0][0]
    
#     # 获取所有边的源节点和目标节点
#     src, dst = graph.edges()
#     print(f'original length{len(src)}')

#     # 使用集合去重 (确保 (src, dst) 和 (dst, src) 都视为相同的边)
#     unique_edges = set(zip(src.tolist(), dst.tolist()))
#     print(f'unique length {len(unique_edges)}')
    
#     # 重新创建图
#     src, dst = zip(*unique_edges)
#     deduplicated_graph = dgl.graph((src, dst), num_nodes=graph.num_nodes())

#     return deduplicated_graph

# # 使用新函数加载图并去重
# for dataset in ['dream5','ecoli','human']:
#     print(dataset)
#     for task in [1,2,3, 4]:
#         for seed in range(10):
#             graph = load_and_deduplicate_graph('./edge_data/%s/whole.graph' % dataset)
#             save_path = './edge_data/%s/' % dataset
#             dp.split_data(args, graph, save_path, seed, task)


import dgl

def load_and_deduplicate_graph(filepath):
    # 加载图
    graph = dgl.load_graphs(filepath)[0][0]
    
    # 获取所有边的源节点和目标节点
    src, dst = graph.edges()
    print(f'Original number of edges: {len(src)}')

    # 使用集合去重，仅去除完全相同的边
    unique_edges = set()
    for s, d in zip(src.tolist(), dst.tolist()):
        if (s, d) not in unique_edges:
            unique_edges.add((s, d))

    print(f'Number of unique edges after deduplication: {len(unique_edges)}')
    
    # 重新创建图
    src, dst = zip(*unique_edges)
    deduplicated_graph = dgl.graph((src, dst), num_nodes=graph.num_nodes())

    return deduplicated_graph

# 使用新函数加载图并去重
for dataset in ['human','dream5','ecoli']:
    print(dataset)
    for task in [1,2,3,4]:
        for seed in range(10):
            graph = load_and_deduplicate_graph('./edge_data/%s/whole.graph' % dataset)
            save_path = './edge_data/%s/' % dataset
            dp.split_data(args, graph, save_path, seed, task)
