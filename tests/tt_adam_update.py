import cProfile
from pstats import Stats

import torch
from tn_gradient.tt import TensorTrain
from tn_gradient.utils import closest_factorization

profiler = cProfile.Profile()
# Set seed
torch.manual_seed(0)

def tt_adam_update(
    gradient: torch.tensor,
    m: TensorTrain,
    v: TensorTrain,
    alpha: float,
    beta1: float,
    beta2: float,
    eps: float
):
    # ranks = gradient.ranks.copy()
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient * gradient)#.round(ranks)
    
    # Replace negative values in the cores of v by 0
    # v.cores = [torch.maximum(core, torch.zeros_like(core)) for core in v.cores]

    m_hat = 1 / (1 - beta1) * m
    v_hat = 1 / (1 - beta2) * v


    # m_hat = m_hat.reconstruct()
    # v_hat = v_hat.reconstruct()

    return alpha * m_hat #/ (torch.sqrt(v_hat) + eps)

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



M, N = 8*8*8, 8*8*8
rank = 6
galore_rank = 64

################## Base tensor ##################

# grad = torch.randn(M, N).abs().to("cuda")
# grad = torch.ones(M, N).to("cuda")
# Create a dummy rank 10 tensor of shape (M, N)
grad = torch.arange(M * N).reshape(M, N).float().to("cuda")
# Print the rank
print("Rank of the tensor: ", torch.linalg.matrix_rank(grad))
m = torch.zeros(M, N).to("cuda")
v = torch.zeros(M, N).to("cuda")

n_params = grad.numel() + m.numel() + v.numel()
print("Total number of parameters: ", n_params)

################## Tensor Train ##################

order = 3
ranks = [1] + [rank] * (order - 1) + [1]
input_shape = closest_factorization(M, order)[0]
output_shape = closest_factorization(N, order)[0]

print("Input shape: ", input_shape)
print("Output shape: ", output_shape)

tt_grad = TensorTrain.from_tensor(
    grad.reshape(input_shape + output_shape),
    ranks=ranks
).to("cuda")
tt_m = TensorTrain.zeros(ranks, input_shape, output_shape).to("cuda")
tt_v = TensorTrain.zeros(ranks, input_shape, output_shape).to("cuda")
n_tt_params = tt_grad.numel() + tt_m.numel() + tt_v.numel()
print("Total number of parameters (TT): ", n_tt_params)
print("Reduction factor: ", n_params / n_tt_params * 100, "%")

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

# Compute the elapsed time for the TT update
import time

start = time.time()
def loop_tt_update():
    for _ in range(50):
        tt_update = tt_adam_update(tt_grad, tt_m, tt_v, alpha, beta1, beta2, eps)
    
profiler.runcall(loop_tt_update)
end = time.time()
print("Elapsed time (TT): ", end - start)

start = time.time()
for _ in range(50):
    update = adam_update(grad, m, v, alpha, beta1, beta2, eps)
end = time.time()
print("Elapsed time (standard): ", end - start)

# start = time.time()
# galore_update = adam_update(galore_grad, galore_m, galore_v, alpha, beta1, beta2, eps) @ V[:galore_rank, :]
# end = time.time()
# print("Elapsed time (GaLore): ", end - start)

tt_update = tt_adam_update(tt_grad, tt_m, tt_v, alpha, beta1, beta2, eps)
tt_update = tt_update.reconstruct()
tt_update = tt_update.reshape(M, N)

# Check if NaN in tt_update
# print("NaN in TT update: ", torch.isnan(tt_update).any())
# # Print the positions of NaN in tt_update
# print("Positions of NaN in TT update: ", torch.where(torch.isnan(tt_update)))

print("TT update: ", tt_update)
# print("Galore update: ", galore_update)
# print("Update: ", update)

print("L2 Norm (TT): ", torch.linalg.norm(tt_update - update))
# print("L2 Norm (GaLore): ", torch.linalg.norm(galore_update - update))

# Print norm L1 between the updates
print("L1 Norm (TT):", torch.linalg.norm(tt_update - update, ord=1))
# print("L1 Norm (GaLore):", torch.linalg.norm(galore_update - update, ord=1))

# profiler.print_stats()
profiler.print_stats(sort='tottime')

# A = torch.arange(12).reshape(1, 12).float() / 12
# B = torch.arange(12*8).reshape((12, 8)).float() / 12 / 8
# C = torch.arange(8).reshape((1, 8)).float() / 8

# # print sqrt(ABC)
# D = torch.einsum("ij,jk,lk->il", A, B, C)
# print(D)
# print(torch.sqrt(D))

# print(torch.sqrt(A) @ torch.sqrt(B) @ torch.sqrt(C).t())

# A = torch.arange(2*2).reshape(2, 2).float()
# Q, R = torch.linalg.qr(A, mode="complete")
# Q2, R2 = torch.abs(Q), torch.abs(R)
# print(Q, R)
# print(Q2 @ R2)

# tensor = torch.arange(2*2*2*3*3*3).reshape(2, 2, 2, 3, 3, 3).float()
# tt = TensorTrain.from_tensor(tensor, [1, 3, 3, 1])

# # Print the norm between the tensor and the tensor train
# print("||A - A_tt|| = ", torch.linalg.norm(tensor - tt.reconstruct()).float())

# tt_sqrt = tt.sqrt()
# print("||sqrt(A) - sqrt(A_tt)|| = ", torch.linalg.norm(torch.sqrt(tensor) - tt_sqrt.reconstruct()).float())
