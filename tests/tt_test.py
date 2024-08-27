import torch
from tn_gradient.tt import TensorTrain

A = torch.arange(2*2*2*3*3*3).reshape((2,2,2,3,3,3)).float()
B = torch.arange(2*2*2*3*3*3).reshape((2,2,2,3,3,3)).float()

ttA = TensorTrain.from_tensor(A, [1, 4, 4, 1])
ttB = TensorTrain.from_tensor(B, [1, 4, 4, 1])

# print(ttA.reconstruct())

print(A.sqrt())
print(ttA.sqrt().reconstruct())

# ttC = ttA + ttB
# print(ttC.ranks, ttC.in_shape, ttC.out_shape)
# C_ = ttC.reconstruct()
# print(C_)
# print(A + B)

# ttC = 2 * ttC
# print(ttC.reconstruct())

# # A = torch.arange(2*2*3*3).reshape((2,2,3,3)).float()
# # B = torch.arange(2*2*3*3).reshape((2,2,3,3)).float()

# # ttA = TensorTrain.from_tensor(A, [1, 2, 1])
# # ttB = TensorTrain.from_tensor(B, [1, 2, 1])

# ttD = ttA * ttA
# ttE = ttD.clone()
# print([core.shape for core in ttE.cores])
# print(ttD.ranks)
# # print(ttE.cores[0])
# # ttF = ttE.round([1, 4, 4, 1])
# ttF = ttD.clone().round([1, 4, 4, 1])
# # print(ttD.ranks)
# # print("ok")
# # ttH = ttD.orthogonalize(new_ranks=[1, 4, 4, 1])
# # # print(ttE.cores[0])
# # print([core.shape for core in ttF.cores])
# print(ttF.reconstruct())
# print(ttH.reconstruct())
# print(ttD.reconstruct())
# print(A * A)
# ttG = TensorTrain.from_tensor(ttD.reconstruct(), [1, 3, 3, 1])
# print([core.shape for core in ttG.cores])

# print(ttG.reconstruct())

# tensor = torch.arange(8*8*8*8*5*5*5*5).reshape((8,8,8,8,5,5,5,5)).float()
# Create a non random, not too simple without being an arange tensor of shape (8, 8, 8, 8, 5, 5, 5, 5))
# tensor = torch.arange(8*8*8*8*5*5*5*5).reshape((8,8,8,8,5,5,5,5)).float()
# tensor /= tensor.max() / 1000
# print(tensor.shape.numel())
# # Put a mask of zeros in the middle of the tensor
# # tensor[:, :, 3:5, 3:5, :, :, :, :] = 0

# # tt = TensorTrain.from_tensor(tensor, [1, 2, 2, 2, 1])
# tt = TensorTrain.from_tensor(tensor, [1, 8, 8, 8, 1])
# reconstructed = tt.reconstruct()

# print(tensor[-1,-1,-1,-1,-1,-1,-1,0], reconstructed[-1,-1,-1,-1,-1,-1,0,0])
# # Check that the reconstructed tensor is close to the original tensor
# # assert torch.allclose(tensor, reconstructed, rtol=1e-3, atol=1e-3)