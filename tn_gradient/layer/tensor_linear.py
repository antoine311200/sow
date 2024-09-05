from math import ceil, sqrt

import torch
import torch.nn as nn
from opt_einsum import contract, contract_expression, contract_path

from tn_gradient.tt import TensorTrain

class TensorTrainLinear(nn.Module):

    def __init__(self, in_features, out_features, ranks, bias=True, device=None, type=None):
        super(TensorTrainLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.order = len(ranks) - 1

        self.contract_expr = None
        
        self.in_core_features = ceil(in_features ** (1 / self.order))
        self.out_core_features = ceil(out_features ** (1 / self.order))

        self.tt = TensorTrain.zeros(
            input_shape=[self.in_core_features] * self.order, 
            output_shape=[self.out_core_features] * self.order,
            ranks=ranks,
            device=device
        )
        self.tt.to_params()
        self.tt.type(type)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def to(self, device):
        self.tt.to(device)
        return super().to(device)

    def reset_parameters(self):
        for core in self.tt.cores:
            nn.init.kaiming_uniform_(core, a=sqrt(5))
            
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

            self.tt.type(self.bias.dtype)

    def forward(self, input):
        input_shape = input.shape
        # Pad the input to the correct size
        input = torch.nn.functional.pad(input, (0, self.in_core_features ** self.order - self.in_features), "constant", 0)
        input = input.view(-1, *[self.in_core_features] * self.order)

        if not self.contract_expr:
            einsum = []
            for i, core in enumerate(self.tt.cores):
                einsum.append(core)
                einsum.append([f"r_{i}", f"i_{i+1}", f"o_{i+1}", f"r_{i+1}"])
            einsum.append(input)
            einsum.append(["b"] + [f"i_{i+1}" for i in range(self.order)])
            einsum.append(["b"] + [f"o_{i+1}" for i in range(self.order)])

            self.contract_expr = contract_expression(contract_path(*einsum)[1].eq, *[core.shape for core in self.tt.cores], input.shape)

        # Contract the cores with the input
        output = self.contract_expr(*self.tt.cores, input)

        # Reshape and unpad the output
        output = output.reshape(-1, torch.prod(torch.tensor(output.shape[1:])))
        output = output[:, :self.out_features]
        output = output.reshape(*input_shape[:-1], self.out_features)

        if self.bias is not None:
            output += self.bias

        # print("Output", output.shape)

        return output
    
class ComposedLinear(nn.Module):

    def __init__(self, in_features, out_features, rank, bias=True, composition=None):
        super(ComposedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias



        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features, rank))
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)

        # self.reset_parameters()