import torch
import torch.nn.functional as F

from math import floor, log, sqrt

from opt_einsum import contract
from scipy import linalg

class TensorTrain:

    def __init__(self, ranks, in_shape, out_shape, device=None) -> None:
        self.order = len(ranks) - 1
        self.ranks = ranks
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        self.cores = [None for _ in range(self.order)]
        self.device = device
        #torch.empty((ranks[i], in_shape[i], out_shape[i], ranks[i+1])) for i in range(self.order)]
        # if device:
            # self.to(device)
        # else:
        #     self.device = self.cores[0].device

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
    def from_cores(cores):
        tt = TensorTrain(
            [core.shape[0] for core in cores] + [1],
            [core.shape[1] for core in cores],
            [core.shape[2] for core in cores],
            device=cores[0].device
        )
        tt.cores = cores
        return tt
    
    @staticmethod
    def zeros(ranks, in_shape, out_shape, device="cpu"):
        tt = TensorTrain(ranks, in_shape, out_shape)
        tt.cores = [torch.zeros((ranks[i], in_shape[i], out_shape[i], ranks[i+1])) for i in range(tt.order)]
        tt.to(device)
        return tt
    
    @staticmethod
    def ones(ranks, in_shape, out_shape, device="cpu"):
        tt = TensorTrain(ranks, in_shape, out_shape)
        tt.cores = [torch.ones((ranks[i], in_shape[i], out_shape[i], ranks[i+1])) for i in range(tt.order)]
        tt.to(device)
        return tt

    def numel(self):
        return sum(core.numel() for core in self.cores)
    
    def to(self, device):
        self.device = device
        if self.cores[0].device == device:
            return self

        for core in self.cores:
            core.data = core.data.to(device)
        return self
    
    def clone(self):
        tt = TensorTrain(self.ranks.copy(), self.in_shape, self.out_shape)
        tt.cores = self.cores.copy()
        return tt

    def decompose(self, tensor: torch.Tensor):
        """Decompose a tensor into a tensor train.
        
        The decomposition is performed by iteratives truncated QR decomposition of the tensor.
        At each step, the current tensor is reshaped into a matrix whose rows correspond to the
        left rank, input and output dimensios of the current core.
        A QR decomposition is then performed on this matrix with a truncation to respect the tt-ranks.
        The left matrix is reshaped to the core shape and the right matrix is kept for the next step as
        the remaining tensor elements to decompose further.
        
        Args:
            tensor (torch.Tensor): The tensor to decompose
            
        Returns:
            TensorTrain: The tensor train resulting from the decomposition
        """
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
    
    def round(self, new_ranks=None, inplace=False, like=None):
        if type(new_ranks) == int:
            new_ranks = [1] + [new_ranks] * (self.order - 1) + [1]
        elif not new_ranks and not like:
            new_ranks = [1] + [i*o for i, o in zip(self.in_shape, self.out_shape)] + [1]
        elif like:
            new_ranks = like.ranks

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

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the tensor from the tensor train.
        
        The reconstruction is done by contracting the cores of the tensor train
        one by one on the bond indices.
        The einstein summation notation is:
            C(i1, ..., in, j1, ..., jm) = C1(r0, i1, j1, r1) * ... * Cn(rn-1, in, jn, rn)
        
        Returns:
            torch.Tensor: The reconstructed tensor
        """
        struct = []
        in_axis = [f"in_{i}" for i in range(self.order)]
        out_axis = [f"out_{i}" for i in range(self.order)]
        for i, core in enumerate(self.cores):
            struct.append(core)
            struct.append((f"rank_{i}", f"in_{i}", f"out_{i}", f"rank_{i+1}"))
        struct.append(in_axis+out_axis)
        return contract(*struct)

    def to_tensor(self):
        return self.reconstruct()
    
    def size(self):
        return [core.size() for core in self.cores]
    
    def norm(self, mode="full"):
        """Compute the norm of the tensor train by performing an einsum contraction
        on the TT and its conjugate."""
        return self.inner(self, mode=mode)
    
    def inner(self, other, mode="right"):
        """Compute the inner product between two tensor trains."""
        struct = []

        if mode == "full":
            for i, (core1, core2) in enumerate(zip(self.cores, other.cores)):
                struct.append(core1)
                struct.append((f"rank_{i}", f"in_{i}", f"out_{i}", f"rank_{i+1}"))
                struct.append(core2)
                struct.append((f"rank2_{i}", f"in_{i}", f"out_{i}", f"rank2_{i+1}"))
        elif mode == "right":
            struct.append(self.cores[-1])
            struct.append((f"rank_{self.order-1}", f"in_{self.order-1}", f"out_{self.order-1}", f"rank_{self.order}"))
            struct.append(other.cores[-1])
            struct.append((f"rank_{self.order-1}", f"in_{self.order-1}", f"out_{self.order-1}", f"rank2_{self.order}"))
        return float(contract(*struct).squeeze())

    def sqrtinv(self, threshold=1e-8, max_iter=4):
        """Compute the element-wise reciprocal of the square root of the tensor train
        using the Newton method to the equation (1/x^2 - y = 0)"""

        # Find the maximum value of the tensor train
        # Then, compute the number of bits to shift the tensor train to the right
        # This allows the square root algorithm to converge
        max_value = float(max([core.abs().max() for core in self.cores]))
        k = floor(log(max_value) / log(4))
        c, revc = (1/(4**k)), (1/2**(k-1))

        # Scale the tensor train by 1/4^k
        A = c * self.clone()
        max_ranks = [1] + [i*o for i, o in zip(self.in_shape, self.out_shape)] + [1]

        while max_iter > 0:
            B = -1/2 * (self * (A * A).round(max_ranks)).add_(-3)
            B = B.round(max_ranks)
            C = A * B
            C = C.round(max_ranks)

            if threshold:
                norm = abs((C - A).norm())
                if norm < threshold:
                    return revc * C
            A = C
            max_iter -= 1

        # Scale back the tensor train by 2^(k-1) to account for the initial scaling
        A = revc * A
        return A

    def sqrt(self, threshold=1e-3):
        """Compute the element-wise square root of the tensor train
        using an iterative method.
        This is a SLOW method and should be used with caution.
        """

        # Find the maximum value of the tensor train
        # Then, compute the number of bits to shift the tensor train to the right
        # This allows the square root algorithm to converge
        # max_value = float(max([core.abs().max() for core in self.cores]))
        max_value = float(self.cores[-1].abs().max())
        k = floor(log(max_value) / log(4))

        # Scale the tensor train by 1/4^k
        A = (1/(4**k)) * self.clone()
        C = A.clone().add_(-1)

        ranks = A.ranks.copy()
        max_iter = 10
        while max_iter > 0 and (A - C).norm() > threshold:
            B = A - 1/2 * (A * C)
            B = B.round(ranks)
            D = 1/4 * (C * C).round(ranks) * (C.add_(-3))
            D = D.round(ranks)
            max_iter -= 1
            A, C = B, D

        # Scale back the tensor train by 2^k to account for the initial scaling
        A = 2**k * A
        return A

    def add_(self, constant):
        """Add a constant to the tensor train, inplace.
        
        An element-wise addition is performed on the cores of the tensor train
        with a corrected constant to account for the number of summed elements
        as there is r0 * r1 * ... * rn sums performed to reconstruct the tensor.

        Args:
            constant (float): The constant to add
            
        Returns:
            TensorTrain: The tensor train with the constant added
        """
        n_inner_params = torch.prod(torch.tensor(self.ranks))
        subconstant = constant / n_inner_params
        is_neg = subconstant < 0
        subconstant = abs(subconstant) ** (1 / self.order)

        cores = []
        for i in range(self.order):
            leftmost = i == 0
            rightmost = i == self.order - 1

            left_matrix = self.cores[i]
            right_matrix = torch.full_like(left_matrix, ((-1 if is_neg else 1) * subconstant))

            if leftmost:
                new_core = torch.cat((left_matrix, right_matrix), dim=-1)
            elif rightmost:
                new_core = torch.cat((left_matrix.transpose(-2, -1), right_matrix.transpose(-2, -1)), dim=0).transpose(-2, -1)
            else:
                left_matrix_padded = F.pad(left_matrix, (0, self.ranks[i+1], 0, 0))
                right_matrix_padded = F.pad(right_matrix, (self.ranks[i], 0, 0, 0))
                new_core = torch.cat([left_matrix_padded, right_matrix_padded], dim=0)

            cores.append(new_core)
        return TensorTrain.from_cores(cores)
        # return self + subconstant * TensorTrain.ones(self.ranks, self.in_shape, self.out_shape, device=self.device)

    def __add__(self, other):
        """Add two tensor trains element-wise.

        The addition of two tensor trains results in a new tensor train
        whose cores are defined by:
            C_k(i_k, j_k) = block_diag(A_k(i_k, j_k), B_k(i_k, j_k))
        with the extremal cores being the concatenation of the extremal cores
        of the two tensor trains along the correct axis.

        In this implementation, we limited the use of for loops on the physical
        dimensions of the tensor train for performance reasons.
        Thus, the element-wise addition for each core is summed up as padding
        the cores along the right axis (for the non-extremal cores) and concatenating
        them.
        
        Args:
            other (TensorTrain): The tensor train to add
            
        Returns:
        TensorTrain: The tensor train resulting from the element-wise addition
        """
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

    def __sub__(self, other):
        """Subtract two tensor trains element-wise."""
        return self + (-1) * other   

    def __rmul__(self, constant):
        """Multiply a tensor train by a constant.
        
        In order to get a homogeneous multiplication, this implementation multiplies
        each core of the tensor train by the constant to the power of 1/d where d is the
        order of the tensor train.
        The counter part of this operation is that it destroys the orthogonality of the tensor train.
        However, the orthogonality is not a property that we need to preserve in this work.
        
        Args:
            constant (float): The constant to multiply by
            
        Returns:
            TensorTrain: The tensor train resulting from the multiplication
        """
        is_neg = constant < 0
        constant = abs(constant)
        subconstant = constant ** (1 / self.order)
        cores = [core * ((-1 if is_neg else 1) * subconstant) for core in self.cores]
        return TensorTrain.from_cores(cores)

    def __mul__(self, other):
        """Multiply two tensor trains element-wise.

        The element-wise multiplication of two tensor trains results in a new tensor train
        whose cores are defined by the dot product of the aligned cores.
            C_k(i_k, j_k) = A_k(i_k, j_k) ⊗ B_k(i_k, j_k)
        To avoid for loops, an einsum contraction is performed directly on the whole cores along
        the correct axes followed by a reshape to get the new core shape.
            
        Args:
            other (TensorTrain): The tensor train to multiply by

        Returns:
            TensorTrain: The tensor train resulting from the element-wise multiplication
        """
        cores = []
        for i in range(self.order):
            left_matrix = self.cores[i]
            right_matrix = other.cores[i]

            new_core = torch.einsum('aijb,cijd->acijbd', left_matrix, right_matrix)
            new_core = new_core.reshape(
                left_matrix.size(0) * right_matrix.size(0),
                left_matrix.size(1),
                left_matrix.size(2),
                left_matrix.size(3) * right_matrix.size(3)
            )

            cores.append(new_core)
        return TensorTrain.from_cores(cores)
    
    def reciprocal(self):
        """Divide two tensor trains element-wise."""
        cores = []
        for i in range(self.order):
            new_core = torch.zeros_like(self.cores[i])
            for inp in range(self.in_shape[i]):
                for out in range(self.out_shape[i]):
                    if i == 0:
                        new_core[:, inp, out, :] = self.cores[i][:, inp, out, :]
                    elif i == self.order - 1:
                        new_core[:, inp, out, :] = self.cores[i][:, inp, out, :]
                    else:
                        new_core[:, inp, out, :] = torch.linalg.inv(self.cores[i][:, inp, out, :])
            cores.append(new_core)
        return TensorTrain.from_cores(cores)
       
    def left_matrix(self, index):
        """Left matrizification of the core at index. C(rk-1, ik, jk, rk) -> C(rk-1 * ik * jk, rk)"""
        return self.cores[index].reshape((self.ranks[index] * self.in_shape[index] * self.out_shape[index], -1))
    
    def right_matrix(self, index):
        """Right matrizification of the core at index. C(rk, ik, jk, rk+1) -> C(rk, ik * jk * rk+1)"""
        return self.cores[index].reshape((-1, self.in_shape[index] * self.out_shape[index] * self.ranks[index+1]))
    
    def to_core(self, matrix, index):
        """Reshape a matrix to the core shape at index C(rk, ik, jk, rk+1)"""
        return matrix.reshape((self.ranks[index], self.in_shape[index], self.out_shape[index], self.ranks[index+1]))