import torch
import torch.nn.functional as F
from math import ceil

from opt_einsum import contract


def pad_matrix(matrix, new_shape):
    # pad_shape = [0, new_shape[1] - matrix.shape[1], 0, new_shape[0] - matrix.shape[0]]
    # padded_matrix = F.pad(matrix, pad_shape)
    # return padded_matrix
    padded_matrix = torch.zeros(new_shape, device=matrix.device)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

def unpad_matrix(matrix, shape):
    return matrix[:shape[0], :shape[1]]

def closest_factorization(n, d):
    factors = []
    p, o = 1, n
    while n > 1:
        k = ceil(n**(1/d))
        factors.append(k)
        n, p, d = n // k, p * k, d - 1
        if n == 1:
            if p < o:
                factors[-1] += n
            return factors, p
        
def generate_rank_k(shape, rank, mix=1, pos=False):
    tensor = torch.zeros(shape)
    for j in range(mix):
        factors = [torch.rand(dim, rank) for dim in shape]
        if not pos:
            factors = [2 * factor - 1 for factor in factors]
        struct = []
        for i, factor in enumerate(factors):
            struct.append(factor)
            struct.append([f"l_{i}", "k"])
        tensor += contract(*struct)
    return tensor

def unfolding(tensor, mode):
    """Unfolds a d-dimensional tensor (a_1, ..., a_d) into a matrix (a_i, a_{i+1} * ... * a_d * a_1 * ... * a_{i-1})"""
    shape = tensor.shape
    d = len(shape)

    if mode < 0:
        mode = d + mode

    if mode < 0 or mode >= d:
        raise ValueError("Mode must be between 1 - d and d + 1, d being the number of dimensions of the tensor")
    
    # Compute the new shape of the unfolded tensor
    # new_shape = (shape[mode], shape[:mode].numel() * shape[mode+1:].numel())
    new_shape = (shape[mode], -1)
    # Permute the dimensions of the tensor to match the new shape
    permuted_tensor = torch.moveaxis(tensor, mode, 0)
    # Reshape the tensor to match the new shape
    reshaped_tensor = torch.reshape(permuted_tensor, new_shape)

    return reshaped_tensor

def left_unfolding(tensor):
    """Unfolds a d-dimensional tensor (a_1, ..., a_d) into a matrix (a_1 * ... * a_{d-1}, a_d)"""
    return unfolding(tensor, -1).t()

def right_unfolding(tensor):
    """Unfolds a d-dimensional tensor (a_1, ..., a_d) into a matrix (a_d, a_1 * ... * a_{d-1})"""
    return unfolding(tensor, 0)

if __name__ == "__main__":
    print(closest_factorization(100, 3))
    print(closest_factorization(1376, 6))
    print(closest_factorization(512, 5))

    tensor = torch.randn(2, 3, 5, 7, 11)
    print(left_unfolding(tensor).shape) # (2*3*5*7, 11)
    print(right_unfolding(tensor).shape) # (2, 3*5*7*11)