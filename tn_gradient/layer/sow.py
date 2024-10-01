import torch
import torch.nn as nn

from opt_einsum import contract_expression, contract_path

from math import sqrt
from typing import List

from dataclasses import dataclass

@dataclass
class SoWArgs:
    device: str = None
    dtype: torch.dtype = None

    rank: int = 16
    n_iter: int = 5
    accumulation_steps: int = 200

class SoWParameter(nn.ParameterList):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_iter: int = 5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SoWParameter, self).__init__(
            [
                nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
                for _ in range(n_iter)
            ]
        )
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter

        self.shape = (self.n_iter, self.in_features, self.out_features)

    def from_weights(self, weights: List[torch.Tensor]) -> None:
        for i, weight in enumerate(weights):
            self[i].data = weight.data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, n_iter={self.n_iter})"


class SoWLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 16,
        n_iter: int = 5,
        accumulation_steps: int = 200,
        init_method: str = "zero_kaiming",
        buffer_proj: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SoWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter
        self.rank = rank
        self.step = 0
        self.virtual_rank = min(rank * n_iter, in_features, out_features)
        self.buffer_proj = buffer_proj

        self.accumulate_every = 2 * accumulation_steps # Account for the forward and backward pass
        # self.accumulated_weight = None
        self.acc_upweight = None
        self.acc_downweight = None

        self.init_method = init_method

        self.downscale_weights = SoWParameter(
            in_features, rank, n_iter=n_iter, device=device, dtype=dtype
        )
        self.upscale_weights = SoWParameter(
            rank, out_features, n_iter=n_iter, device=device, dtype=dtype
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.buffer_proj:
            self.register_buffer("proj_row", torch.eye(self.out_features))
            self.register_buffer("proj_col", torch.eye(self.in_features))

        self.reset_parameters()

        # Add a hook to step the model at each backward pass
        self.register_full_backward_hook(self._step)

    def _step(self, module, grad_input, grad_output):
        self.step += 1

    def reset_parameters(self, reset_scale=1.0) -> None:

        for i, (downscale_weight, upscale_weight) in enumerate(
            zip(self.downscale_weights, self.upscale_weights)
        ):
            if i / self.n_iter >= 1 - reset_scale:
                nn.init.kaiming_uniform_(downscale_weight, a=sqrt(5))
                nn.init.kaiming_uniform_(upscale_weight, a=sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.downscale_weights[0])
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.upscale_weights[0])
            fan_in = (fan_in + fan_in2) / 2
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.step % self.accumulate_every == 0 and self.step > 0:
            self.accumulate()

        out = None
        if self.acc_downweight is not None and self.acc_upweight is not None:
            out = (x @ self.acc_downweight) @ self.acc_upweight
        elif self.acc_downweight is not None and self.acc_upweight is None:
            out = x @ self.acc_downweight

        for downscale_weight, upscale_weight in zip(
            self.downscale_weights, self.upscale_weights
        ):
            output = x @ downscale_weight
            output = output @ upscale_weight
            if out is not None:
                out += output
            else:
                out = output

        if self.bias is not None:
            out += self.bias

        return out
    
    def accumulate(self):

        # Accumulate the weights
        accumalation = torch.sum(torch.stack([
            a.detach() @ b.detach() 
            for a, b in zip(self.downscale_weights, self.upscale_weights)
        ]), dim=0).detach()

        # Compute the full W_acc matrix from the previous accumulated weights
        if self.acc_downweight is not None and self.acc_upweight is not None:
            accumalation = accumalation.to(self.acc_upweight.device) + self.acc_downweight @ self.acc_upweight
        elif self.acc_downweight is not None and self.acc_upweight is None:
            accumalation = accumalation.to(self.acc_downweight.device) + self.acc_downweight

        # Perform QR decomposition to get the new accumulated weights
        # only if the virtual rank is less than the full-rankness
        if self.virtual_rank < min(self.in_features, self.out_features):
            # Convertion to float is necessary for the QR decomposition
            # as CUDA does not support QR decomposition for half precision
            convertion = False
            if accumalation.dtype != torch.float:
                convertion = True

                weight_type = accumalation.dtype
                weight_device = accumalation.device
                accumalation = accumalation.to(torch.float)

            Q, R = torch.linalg.qr(accumalation)
            Q = Q[:, :self.virtual_rank]
            R = R[:self.virtual_rank, :]

            self.acc_downweight, self.acc_upweight = Q, R
            self.acc_downweight.requires_grad = False
            self.acc_upweight.requires_grad = False
            
            if convertion:
                self.acc_downweight = self.acc_downweight.to(weight_device).type(weight_type)
                self.acc_upweight = self.acc_upweight.to(weight_device).type(weight_type)

            self.virtual_rank = min(self.virtual_rank + self.rank * self.n_iter, self.in_features, self.out_features)
            torch.cuda.empty_cache()
        else:
            self.acc_downweight = accumalation
            self.acc_downweight.requires_grad = False
            self.acc_upweight = None
            torch.cuda.empty_cache()

        # Reset weights
        new_downscale_weights = [torch.zeros_like(x) for x in self.downscale_weights]
        new_upscale_weights = [torch.zeros_like(x) for x in self.upscale_weights]

        # Change initializations with random canceling
        for i in range(0, self.n_iter):
            if self.init_method == "kaiming_kaiming":
                nn.init.kaiming_uniform_(new_upscale_weights[i], a=sqrt(5))
            nn.init.kaiming_uniform_(new_downscale_weights[i], a=sqrt(5))

        self.downscale_weights.from_weights(new_downscale_weights)
        self.upscale_weights.from_weights(new_upscale_weights)

    def project_grad(self):
        """Use the buffered projector onto the orthogonal complement of the accumulated weight
        rowspace and colspace on the gradient of the downscale and upscale weights."""
        if self.buffer_proj:
            with torch.no_grad():
                for i in range(0, self.n_iter):
                    if self.downscale_weights[i].data.grad is not None:
                        self.downscale_weights[i].data.grad = self.downscale_weights[i].data.grad @ self.proj_col
                        self.upscale_weights[i].data.grad = self.proj_row @ self.upscale_weights[i].data.grad

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, n_iter={self.n_iter}"


from math import ceil, sqrt


class TensorTrainParameter(nn.ParameterList):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        order: int,
        n_iter: int = 5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        ranks = [1] + [rank] * (order - 1) + [1]
        in_subfeatures = ceil(in_features ** (1 / order))
        out_subfeatures = ceil(out_features ** (1 / order))

        super(TensorTrainParameter, self).__init__([nn.ParameterList([
            nn.Parameter(
                torch.empty(
                    ranks[i],
                    in_subfeatures,
                    out_subfeatures,
                    ranks[i + 1],
                    **factory_kwargs,
                )
            )
            for i in range(order)]) for _ in range(n_iter)
        ])
        self.ranks = ranks
        self.rank = rank
        self.order = order
        self.in_subfeatures = in_subfeatures
        self.out_subfeatures = out_subfeatures
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, order={self.order}, rank={self.rank}, n_iter={self.n_iter})"


class SumTTLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        outer_rank: int = 16,
        inner_rank: int = 16,
        order: int = 4,
        n_iter: int = 5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SumTTLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter
        self.order = order
        self.outer_rank = outer_rank
        self.inner_rank = inner_rank

        self.downscale_weights = TensorTrainParameter(
            in_features, outer_rank**order, inner_rank, order, n_iter=n_iter, **factory_kwargs
        )
        self.upscale_weights = TensorTrainParameter(
            outer_rank**order, out_features, inner_rank, order, n_iter=n_iter, **factory_kwargs
        )
        self.contraction_expr = None

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self, reset_scale=1.0) -> None:

        for i, (downscale_weight, upscale_weight) in enumerate(
            zip(self.downscale_weights, self.upscale_weights)
        ):
            if i / self.n_iter >= 1 - reset_scale:
                for core in downscale_weight:
                    nn.init.kaiming_uniform_(core, a=sqrt(5))
                        
                for core in upscale_weight:
                    nn.init.kaiming_uniform_(core, a=sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.downscale_weights[0][0])
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.upscale_weights[0][0])
            fan_in = (fan_in + fan_in2) / 2
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad and reshape the last dimension into n=oder axis of shape in_subfeatures
        original_shape = x.shape
        x = torch.nn.functional.pad(x, (0, self.downscale_weights.in_subfeatures ** self.order - self.in_features), "constant", 0)
        x = x.reshape(*original_shape[:-1], *[self.downscale_weights.in_subfeatures] * self.order)

        out = None
        for downscale_weight, upscale_weight in zip(
            self.downscale_weights, self.upscale_weights
        ):
            if self.contraction_expr is None:
                einsum = []
                shapes = []
                for i, core in enumerate(downscale_weight):
                    einsum.append(core)
                    einsum.append([f"r_{i}", f"i_{i+1}", f"m_{i+1}", f"r_{i+1}"])
                    shapes.append(core.shape)
                for i, core in enumerate(upscale_weight):
                    einsum.append(core)
                    einsum.append([f"s_{i}", f"m_{i+1}", f"o_{i+1}", f"s_{i+1}"])
                    shapes.append(core.shape)
                einsum.append(x)
                einsum.append(["batch", "token"] + [f"i_{i+1}" for i in range(self.order)])
                einsum.append(["batch", "token"] + [f"o_{i+1}" for i in range(self.order)])
                shapes.append(x.shape)

                self.contraction_expr = contract_expression(contract_path(*einsum)[1].eq, *shapes)

            output = self.contraction_expr(*downscale_weight, *upscale_weight, x)
            output = output.reshape(-1, torch.prod(torch.tensor(output.shape[2:])))
            output = output[:, :self.out_features]
            output = output.reshape(*original_shape[:-1], self.out_features)

            if out is not None:
                out += output
            else:
                out = output

        if self.bias is not None:
            out += self.bias

        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, inner_rank={self.inner_rank}, outer_rank={self.outer_rank} n_iter={self.n_iter}"
