import torch
import dgl
print("cuda" if torch.cuda.is_available() else "cpu")