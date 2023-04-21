__all__ = ["coo_2_torch_tensor", "unsorted_segment_sum", "unsorted_segment_mean", "compute_centroids"]
from torch import nn
import torch
import numpy as np

def coo_2_torch_tensor(sparse_mx, sparse=True):
    """
    Args :
        scipy matrix
        sparse: boolean value specifying if the matrix is sparse or not
    Returns :
        a torch tensor
    """

    if sparse:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    else:
        return torch.FloatTensor(sparse_mx.todense())

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
def compute_centroids(x, b, simplex_dim = 2):
    b_i, b_j = b
    x_i = x.index_select(-2, b_i)
    num_simplices = int(b.shape[1]/simplex_dim)
    #print(num_simplices)# The number of non-zero entries equals the number of upper simplices * dimension of boundary (edges have 2 boundary vertices, triangles have 3 boundary edges, etc.)
    x_j = unsorted_segment_mean(x_i, b_j, num_simplices) #Take the mean of the x_i values for each simplex.
    return x_j