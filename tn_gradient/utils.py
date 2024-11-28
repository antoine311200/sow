import torch
import torch.nn.functional as F
from math import ceil

from opt_einsum import contract
from scipy.stats import ortho_group

def qr_weight(weight: torch.tensor, rank: int = None):
    # Convertion to float is necessary for the QR decomposition
    # as CUDA does not support QR decomposition for half precision
    convertion = False
    weight_device = weight.device
    if weight.dtype != torch.float:
        convertion = True

        weight_type = weight.dtype
        weight = weight.to(torch.float)

    Q, R = torch.linalg.qr(weight)
    if rank:
        Q = Q[:, :rank]
        R = R[:rank, :]

    Q = Q.to(weight_device)
    R = R.to(weight_device)
    if convertion:
        Q = Q.type(weight_type)
        R = R.type(weight_type)

    return Q, R

def svd_weight(weight: torch.tensor, rank: int = None):
    # Convertion to float is necessary for the SVD decomposition
    # as CUDA does not support QR decomposition for half precision
    convertion = False
    weight_device = weight.device
    if weight.dtype != torch.float:
        convertion = True

        weight_type = weight.dtype
        weight = weight.to(torch.float)

    U, S, V = torch.linalg.svd(weight)
    if rank:
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]

    U = U.to(weight_device)
    S = S.to(weight_device)
    V = V.to(weight_device)
    if convertion:
        U = U.type(weight_type)
        S = S.type(weight_type)
        V = V.type(weight_type)

    return U, S, V

def randhaar(n):
    """Generates a random n x n orthogonal matrix Q with Haar distribution."""
    Q = torch.tensor(ortho_group.rvs(dim=n), dtype=torch.float32)
    return Q

def randuptri(n, scale=1.0):
    """Generates a random upper triangular matrix R with scaled chi-distributed diagonal entries."""
    R = torch.triu(torch.randn(n, n))
    for i in range(n):
        # Set each diagonal element as a scaled chi-distributed value
        R[i, i] = torch.sqrt(torch.distributions.Chi2(df=n-i).sample()) * scale
    return R

def perturbe_random(matrix: torch.tensor, scale=0.02):
    """Perturbe a matrix by a random gaussian noise"""
    noise = torch.randn(matrix.size(), device=matrix.device) * scale
    perturbed_matrix = matrix + noise
    return perturbed_matrix

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

from termcolor import colored

def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
def __colorized_str__(self, indent=2):
    """
    Modify the __str__ method of a PyTorch module to color-code submodules and group repeated layers.

    - Red: Modules with all parameters requiring no gradients.
    - Green: Modules with all parameters trainable (require gradients).
    - Orange: Modules with a mix of trainable and non-trainable parameters.
    """

    def get_color(module):
        """Determine the color for a module based on parameter trainability."""
        params = list(module.parameters(recurse=True))
        if not params:
            return "white"
        elif all(not p.requires_grad for p in params):
            return "red"
        elif all(p.requires_grad for p in params):
            return "green"
        return "yellow"

    def summarize_repeated_modules(lines):
        """
        Group repeated submodules into a single summary line if they have the same color and type.
        """
        if len(lines) == 0: return []
        # previous_key, previous_module_str, previous_color = lines[0]
        grouped_lines = []
        previous_key, previous_module_str, previous_color = None, None, None
        count = 0
        start_index = None

        for i, (key, mod_str, color) in enumerate(lines):
            if mod_str == previous_module_str and color == previous_color and (previous_key == key or key.isdigit()):
                if count == 0: start_index = i - 1
                count += 1
            else:
                if count > 0:
                    # Summarize the group
                    grouped_lines.pop()
                    grouped_lines.append(
                        (f"{start_index}-{i-1}", f"{count + 1} x {mod_str}", previous_color)
                    )
                grouped_lines.append((key, mod_str, color))
                count = 0
            previous_module_str = mod_str
            previous_color = color
            previous_key = key

        # # Handle the last group
        if count > 0:
            grouped_lines.pop()
            grouped_lines.append(
                (f"{start_index}-{len(lines)-1}", f"{count + 1} x {mod_str}", previous_color)
            )

        return grouped_lines

    extra_lines = []
    extra_repr = self.extra_repr()
    if extra_repr:
        extra_lines = extra_repr.split("\n")

    # Recursive representation for child modules
    child_lines = []
    for key, module in self._modules.items():
        if module is None:
            continue
        color = get_color(module)
        mod_str = __colorized_str__(module, indent=indent + 2)
        child_lines.append((key, mod_str, color))

    # Summarize repeated child modules
    summarized_lines = summarize_repeated_modules(child_lines)

    # Build the string for the current module
    child_lines_formatted = []
    for key, mod_str, color in summarized_lines:
        colored_key = colored(f"({key}):", color)
        child_lines_formatted.append(_addindent(f"{colored_key} {mod_str}", 2))

    lines = extra_lines + child_lines_formatted
    main_str = self._get_name()
    if lines:
        if len(extra_lines) == 1 and not child_lines_formatted:
            main_str += "(" + extra_lines[0] + ")"
        else:
            main_str += "(\n  " + "\n  ".join(lines) + "\n)"

    return main_str