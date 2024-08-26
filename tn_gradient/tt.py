import torch
import torch.nn.functional as F

from opt_einsum import contract
from scipy import linalg

class TensorTrain:

    def __init__(self, ranks, in_shape, out_shape, device=None) -> None:
        self.order = len(ranks) - 1
        self.ranks = ranks
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.device = device
        
        self.cores = [torch.empty((ranks[i], in_shape[i], out_shape[i], ranks[i+1])) for i in range(self.order)]
        if device:
            for core in self.cores:
                core.data = core.data.to(device)

    @staticmethod
    def from_tensor(tensor: torch.Tensor, ranks: list):
        """Assuming tensor axis are of the form (*in_shape, *out_shape) 
                (e.g (2, 2, 2, 3, 3, 3) for a tensor_(2, 2, 2)^(3, 3, 3))
        """
        tt = TensorTrain(ranks, tensor.shape[:len(tensor.shape)//2], tensor.shape[len(tensor.shape)//2:])
        # Alternate the in and out shapes of the tensor
        tensor = tensor.permute(*[i for pair in zip(range(tt.order), range(tt.order, 2*tt.order)) for i in pair])
        tt.decompose(tensor)
        return tt

    @staticmethod
    def from_matrix(matrix, rank, order):
        ranks = [1] + [rank] * (order - 1) + [1]

    @staticmethod
    def from_cores(cores):
        tt = TensorTrain([core.shape[0] for core in cores] + [1], [core.shape[1] for core in cores], [core.shape[2] for core in cores])
        tt.cores = cores
        return tt
    
    @staticmethod
    def zeros(ranks, in_shape, out_shape):
        return TensorTrain(ranks, in_shape, out_shape)

    def numel(self):
        return sum(core.numel() for core in self.cores)
    
    def to(self, device):
        for core in self.cores:
            core.data = core.data.to(device)
        return self
    
    def clone(self):
        tt = TensorTrain(self.ranks.copy(), self.in_shape.copy(), self.out_shape.copy())
        tt.cores = [core.clone() for core in self.cores]
        return tt

    def decompose(self, tensor: torch.Tensor):
        """Decompose a tensor into a tensor train"""
        for k in range(self.order - 1):
            L = tensor.reshape(self.ranks[k] * self.in_shape[k] * self.out_shape[k], -1)
            Q, R = torch.linalg.qr(L, mode="complete")
            
            right_rank = self.ranks[k+1]
            Q = Q[:, :right_rank]
            R = R[:right_rank, :]
            
            self.cores[k] = torch.reshape(Q, (self.ranks[k], self.in_shape[k], self.out_shape[k], right_rank))
            tensor = R

        self.cores[-1] = torch.reshape(tensor, (self.ranks[-2], self.in_shape[-1], self.out_shape[-1], self.ranks[-1]))
    
    def orthogonalize(self, mode="left", new_ranks=None, inplace=False):
        if inplace:
            if mode == "left":
                for k in range(self.order - 1):
                    L = self.left_matrix(k)
                    R = self.right_matrix(k+1)


                    Q, S = torch.linalg.qr(L)
                    W = S @ R

                    if new_ranks:
                        Q = Q[:, :new_ranks[k]]
                        W = W[:new_ranks[k], :]
                    
                    self.ranks[k+1] = Q.shape[1]
                    self.cores[k] = self.to_core(Q, k)
                    self.cores[k+1] = self.to_core(W, k+1)
            elif mode == "right":
                for k in range(self.order - 1, 0, -1):
                    L = self.left_matrix(k-1)
                    R = self.right_matrix(k)

                    Q, S = torch.linalg.qr(R.t())
                    W = L @ S.t()
                    
                    if new_ranks:
                        Q = Q[:, :new_ranks[k]]
                        W = W[:new_ranks[k], :]
                        self.ranks[k] = new_ranks[k]

                    self.ranks[k] = W.shape[1]
                    self.cores[k-1] = self.to_core(W, k-1)
                    self.cores[k] = self.to_core(Q.t(), k)
            return self
        else:
            tt = self.clone()
            tt.orthogonalize(mode, new_ranks, inplace=True)
            return tt
    
    def round(self, new_ranks, inplace=False):
        if inplace:
            self.orthogonalize(mode="right", inplace=True)

            for k in range(self.order - 1):
                L = self.left_matrix(k)
                R = self.right_matrix(k+1)

                Q, S = torch.linalg.qr(L, mode="complete")
                Q = Q[:, :new_ranks[k+1]]
                S = S[:new_ranks[k+1], :]
                W = S @ R

                self.ranks[k] = new_ranks[k]
                self.ranks[k+1] = new_ranks[k+1]
                
                self.cores[k] = self.to_core(Q, k)
                self.cores[k+1] = self.to_core(W, k+1)
            return self
        else:
            tt = self.clone()
            tt.round(new_ranks, inplace=True)
            return tt

    def reconstruct(self):
        """Reconstruct the tensor from the tensor train"""
        struct = []
        in_axis = [f"in_{i}" for i in range(self.order)]
        out_axis = [f"out_{i}" for i in range(self.order)]
        for i, core in enumerate(self.cores):
            struct.append(core)
            struct.append((f"rank_{i}", f"in_{i}", f"out_{i}", f"rank_{i+1}"))
        struct.append(in_axis+out_axis)
        return contract(*struct)

    def __add__(self, other):
        cores = []
        for i in range(self.order):
            leftmost = i == 0
            rightmost = i == self.order - 1

            left_matrix = self.cores[i]
            right_matrix = other.cores[i]

            if leftmost:
                new_core = torch.cat((left_matrix, right_matrix), dim=-1)
            elif rightmost:
                new_core = torch.cat((left_matrix.transpose(-2, -1), right_matrix.transpose(-2, -1)), dim=0).transpose(-2, -1)
            else:
                left_matrix_padded = F.pad(left_matrix, (0, other.ranks[i+1], 0, 0))
                right_matrix_padded = F.pad(right_matrix, (self.ranks[i], 0, 0, 0))
                new_core = torch.cat([left_matrix_padded, right_matrix_padded], dim=0)

            cores.append(new_core)

        return TensorTrain.from_cores(cores)        

    def __rmul__(self, constant):
        subconstant = constant ** (1 / self.order)
        cores = [core * subconstant for core in self.cores]
        return TensorTrain.from_cores(cores)

    def __mul__(self, other):
        cores = []
        for i in range(self.order):
            left_matrix = self.cores[i]
            right_matrix = other.cores[i]

            new_core = torch.einsum('aijb,cijd->acijbd', left_matrix, right_matrix)
            new_core = new_core.reshape(left_matrix.size(0) * right_matrix.size(0), left_matrix.size(1), left_matrix.size(2), left_matrix.size(3) * left_matrix.size(3))

            cores.append(new_core)
        return TensorTrain.from_cores(cores)

    def left_matrix(self, index):
        return self.cores[index].reshape((self.ranks[index] * self.in_shape[index] * self.out_shape[index], -1))
    
    def right_matrix(self, index):
        return self.cores[index].reshape((-1, self.in_shape[index] * self.out_shape[index] * self.ranks[index+1]))
    
    def to_core(self, matrix, index):
        return matrix.reshape((self.ranks[index], self.in_shape[index], self.out_shape[index], self.ranks[index+1]))