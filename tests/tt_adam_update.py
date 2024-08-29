import cProfile
from pstats import Stats
from math import ceil, log2

import torch
from opt_einsum import contract

from tn_gradient.tt import TensorTrain
from tn_gradient.utils import closest_factorization, pad_matrix, unpad_matrix

profiler = cProfile.Profile()

# Set seed
torch.manual_seed(0)


def generate_rank_k_tensor(shape, k, sumof=1):
    tensor = torch.zeros(shape)
    for j in range(sumof):
        factors = [2 * torch.rand(dim, k) - 1 for dim in shape]
        struct = []
        for i, factor in enumerate(factors):
            struct.append(factor)
            struct.append([f"l_{i}", "k"])
        tensor += contract(*struct)
    return tensor


def sgd_update(
    gradient: TensorTrain,
    momentum: float,
    alpha: float,
    dampening: float,
    nesterov: bool,
    buffer: TensorTrain = None,
):
    if momentum:
        if buffer is None:
            print("wtf")
            buffer = gradient
        else:
            buffer = momentum * buffer + (1 - dampening) * gradient 
        if nesterov:
            gradient = gradient + momentum * buffer
        else:
            gradient = buffer

    if isinstance(gradient, TensorTrain):
        gradient = gradient.round()

    return alpha * gradient


def tt_adam_update(
    gradient: torch.tensor,
    m: TensorTrain,
    v: TensorTrain,
    alpha: float,
    beta1: float,
    beta2: float,
    eps: float
):

    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient * gradient)#.round(ranks)
    v = v.round(gradient.input_shape[0] * gradient.output_shape[0])

    # Replace negative values in the cores of v by 0
    # v.cores = [torch.maximum(core, torch.zeros_like(core)) for core in v.cores]

    m_hat = (1 / (1 - beta1)) * m
    v_hat = (1 / (1 - beta2)) * v

    # m_hat = m_hat.reconstruct()
    # v_hat = v_hat.reconstruct()

    return alpha * m_hat * v_hat.sqrtinv(max_iter=50, threshold=1e-6)
    # return alpha * m_hat / (v_hat.sqrt().add_(eps))

def adam_update(
    gradient: torch.tensor,
    m: torch.tensor,
    v: torch.tensor,
    alpha: float,
    beta1: float,
    beta2: float,
    eps: float
):
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient * gradient)

    m_hat = 1 / (1 - beta1) * m
    v_hat = 1 / (1 - beta2) * v

    return alpha * m_hat / (torch.sqrt(v_hat) + eps)


################## Parameters ##################

M, N = 3**4, 3**4#8*8*8, 8*8*8
galore_rank = 64
rank = 4

order = 4
ranks = [1] + [rank] * (order - 1) + [1]
input_shape = closest_factorization(M, order)[0]
output_shape = closest_factorization(N, order)[0]

################## Base tensor ##################

# M_padded = 2 ** ceil(log2(M))
# N_padded = 2 ** ceil(log2(N))
# # print("NEXT", N1, N2)
# print(M_padded, N_padded)

mm = ceil(M ** (1 / order))
nn = ceil(N ** (1 / order))
M_padded = mm ** order
N_padded = nn ** order

rank_grad = generate_rank_k_tensor(input_shape+output_shape, 2, sumof=2)
# print("Rank of the tensor: ", torch.linalg.matrix_rank(rank_grad))
grad = rank_grad.reshape(M, N).float().to("cuda")
print("Shape of the gradient: ", grad.shape)
padded_grad = pad_matrix(grad, (M_padded, N_padded))
print("New shape of the gradient: ", grad.shape)

m = torch.zeros(M, N).to("cuda")
v = torch.zeros(M, N).to("cuda")

n_params = grad.numel() + m.numel() + v.numel()
print("Total number of parameters: ", n_params)

################## Tensor Train ##################

print("Input shape: ", input_shape)
print("Output shape: ", output_shape)

padded_input_shape = (mm, ) * order
padded_output_shape = (nn, ) * order
print("Padded input shape: ", padded_input_shape)
print("Padded output shape: ", padded_output_shape)
tt_grad = TensorTrain.from_tensor(
    # grad.reshape(input_shape + output_shape),
    padded_grad.reshape(padded_input_shape + padded_output_shape),
    ranks=ranks
).to("cuda")
tt_m = TensorTrain.zeros(ranks, padded_input_shape, padded_output_shape).to("cuda")
tt_v = TensorTrain.zeros(ranks, padded_input_shape, padded_output_shape).to("cuda")
# tt_m = TensorTrain.zeros(ranks, input_shape, output_shape).to("cuda")
# tt_v = TensorTrain.zeros(ranks, input_shape, output_shape).to("cuda")
n_tt_params = tt_grad.numel() + tt_m.numel() + tt_v.numel()
print("Total number of parameters (TT): ", n_tt_params)
print("Reduction factor: ", n_params / n_tt_params, "times")

# Print decomposition error from the TT decomposition
tt2t = unpad_matrix(tt_grad.reconstruct().reshape(M_padded, N_padded), (M, N))
print("TT decomposition error: ", torch.linalg.norm(tt2t - grad))

################## Galore ##################

# U, S, V = torch.svd(grad)
# galore_grad = (V[:galore_rank, :] @ grad).T
# galore_m = torch.zeros(N, galore_rank)
# galore_v = torch.zeros(N, galore_rank)

# n_galore_params = galore_grad.numel() + galore_m.numel() + galore_v.numel()
# print("Total number of parameters (GaLore): ", n_galore_params)

alpha = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

momentum = 0.9
dampening = 0.0
nesterov = True

# Compute the elapsed time for the TT update
import time

start = time.time()
def loop_tt_update():
    for _ in range(50):
        # tt_update = tt_adam_update(tt_grad, tt_m, tt_v, alpha, beta1, beta2, eps)
        tt_update = sgd_update(tt_grad, momentum, alpha, dampening, nesterov, buffer=tt_m)

profiler.runcall(loop_tt_update)
end = time.time()
print("Elapsed time (TT): ", end - start)

start = time.time()
for _ in range(50):
    # update = adam_update(grad, m, v, alpha, beta1, beta2, eps)
    update = sgd_update(grad, momentum, alpha, dampening, nesterov, buffer=m)
end = time.time()
print("Elapsed time (standard): ", end - start)

# start = time.time()
# galore_update = adam_update(galore_grad, galore_m, galore_v, alpha, beta1, beta2, eps) @ V[:galore_rank, :]
# end = time.time()
# print("Elapsed time (GaLore): ", end - start)

# tt_update = tt_adam_update(tt_grad, tt_m, tt_v, alpha, beta1, beta2, eps)
tt_update = sgd_update(tt_grad, momentum, alpha, dampening, nesterov, buffer=tt_m)
# if isinstance(tt_update, TensorTrain):
#     tt_update = tt_update.reconstruct()
# tt_update = tt_update.reshape(M, N)
tt_update = unpad_matrix(tt_update.reconstruct().reshape(M_padded, N_padded), (M, N))


# print("TT update: ", tt_update)
# print("Galore update: ", galore_update)
# print("Update: ", update)

print("L2 Norm (TT): ", torch.linalg.norm(tt_update - update))
# print("L2 Norm (GaLore): ", torch.linalg.norm(galore_update - update))

# Print norm L1 between the updates
# print("L1 Norm (TT):", torch.linalg.norm(tt_update - update, ord=1))
# print("L1 Norm (GaLore):", torch.linalg.norm(galore_update - update, ord=1))

# profiler.print_stats()
# profiler.print_stats(sort='tottime')