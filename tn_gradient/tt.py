import torch
from opt_einsum import contract
from scipy import linalg

class TensorTrain:

    def __init__(self, ranks, in_shape, out_shape) -> None:
        self.order = len(ranks) - 1
        self.ranks = ranks
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        self.cores = [torch.zeros((ranks[i], in_shape[i], out_shape[i], ranks[i+1])) for i in range(self.order)]

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
            
            new_core = torch.zeros((
                self.ranks[i] + (other.ranks[i] if not leftmost else 0),
                self.in_shape[i], self.out_shape[i],
                self.ranks[i+1] + (other.ranks[i+1] if not rightmost else 0)
            ))

            for inp in range(self.in_shape[i]):
                for out in range(self.out_shape[i]):
                    left_matrix = self.cores[i][:, inp, out, :]
                    right_matrix = other.cores[i][:, inp, out, :]
                    if leftmost:
                        new_core[:, inp, out, :] = torch.cat((left_matrix, right_matrix), dim=-1)
                    elif rightmost:
                        new_core[:, inp, out, :] = torch.cat((left_matrix.t(), right_matrix.t()), dim=1).t()
                    else:
                        new_core[:, inp, out, :] = linalg.block_diag((left_matrix, right_matrix))

            cores.append(new_core)
        return TensorTrain.from_cores(cores)

    def __rmul__(self, constant):
        subconstant = constant ** (1 / self.order)
        cores = [core * subconstant for core in self.cores]
        return TensorTrain.from_cores(cores)

    def __mul__(self, other):
        cores = []
        for i in range(self.order):
            new_core = torch.zeros((
                self.ranks[i] * other.ranks[i],
                self.in_shape[i], self.out_shape[i],
                self.ranks[i+1] * other.ranks[i+1]
            ))

            for inp in range(self.in_shape[i]):
                for out in range(self.out_shape[i]):
                    left_matrix = self.cores[i][:, inp, out, :]
                    right_matrix = other.cores[i][:, inp, out, :]
                    
                    new_core[:, inp, out, :] = torch.kron(left_matrix, right_matrix)

            cores.append(new_core)
        return TensorTrain.from_cores(cores)