import torch
import torch.nn as nn

from opt_einsum import contract_expression, contract_path

from math import sqrt
from typing import Tuple, List

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
        self.buffer_proj = buffer_proj

        self.accumulate_every = accumulation_steps 
        self.accumulated_weight = None

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
            # print("Resetting parameters", i, i/self.n_iter >= 1-reset_scale)
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

        if self.accumulated_weight is not None:
            out = x @ self.accumulated_weight
        else:
            out = None

        # stacked_downscale_weights = torch.stack([w for w in self.downscale_weights]) # Shape: (n, in_features, rank)
        # stacked_upscale_weights = torch.stack([w for w in self.upscale_weights]) # Shape: (n, rank, out_features)

        # # print("Stacked downscale weights:", stacked_downscale_weights.shape)
        # # print("Stacked upscale weights:", stacked_upscale_weights.shape)

        # x_expanded = x.unsqueeze(0).expand(self.n_iter, -1, -1, -1)
        # x_expanded = x_expanded.view(self.n_iter, x.size(0) * x.size(1), -1) # Shape: (n, batch * length, in_features)

        # # print("Expanded input:", x_expanded.shape)

        # intermediate = torch.bmm(x_expanded, stacked_downscale_weights) # Shape: (n, batch * length, rank)
        # intermediate = torch.bmm(intermediate, stacked_upscale_weights) # Shape: (n, batch * length, out_features)

        # intermediate = intermediate.view(self.n_iter, x.size(0), x.size(1), self.out_features) # Shape: (n, batch, length, out_features)
        # intermediate = intermediate.sum(dim=0) # Shape: (batch, length, out_features)

        # if out is not None: out += intermediate
        # else: out = intermediate

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
        accumalation = torch.sum(torch.stack([a.detach() @ b.detach() for a, b in zip(self.downscale_weights, self.upscale_weights)]), dim=0).detach()#.cpu().numpy()

        if self.accumulated_weight is None:
            self.accumulated_weight = accumalation
        else:
            self.accumulated_weight = self.accumulated_weight.to(accumalation.device) + accumalation

        self.accumulated_weight = self.accumulated_weight.detach().to(self.downscale_weights[0].device)
        self.accumulated_weight.requires_grad = False

        if self.buffer_proj:
            W = self.accumulated_weight
            self.proj_row = torch.eye(W.size(1), device=W.device) - W.t() @ torch.inverse(W @ W.t()) @ W
            self.proj_col = torch.eye(W.size(0), device=W.device) - W @ torch.inverse(W.t() @ W) @ W.t()

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
        
        if self.init_method == "kaiming_kaiming":
            random_accumalation = torch.sum(torch.stack([a.detach() @ b.detach() for a, b in zip(self.downscale_weights, self.upscale_weights)]), dim=0).detach()
            self.accumulated_weight = self.accumulated_weight - random_accumalation

    def project_grad(self):
        """Use the buffered projector onto the orthogonal complement of the accumulated weight
        rowspace and colspace on the gradient of the downscale and upscale weights."""
        if self.buffer_proj:
            with torch.no_grad():
                for i in range(0, self.n_iter):
                    if self.downscale_weights[i].data.grad is not None:
                        self.downscale_weights[i].data.grad = self.downscale_weights[i].data.grad @ self.proj_col
                        self.upscale_weights[i].data.grad = self.proj_row @ self.upscale_weights[i].data.grad


    def mingle(self):
        scale = 1 / self.n_iter

        new_downscale_weights = [torch.zeros_like(x) for x in self.downscale_weights]
        new_upscale_weights = [torch.zeros_like(x) for x in self.upscale_weights]
        # new_downscale_weights = [torch.zeros(self.n_iter, self.in_features, self.rank).to(self.downscale_weights[0].device).type(self.downscale_weights[0].dtype) for _ in range(self.n_iter)]
        # new_upscale_weights = [torch.zeros(self.n_iter, self.rank, self.out_features).to(self.upscale_weights[0].device).type(self.upscale_weights[0].dtype) for _ in range(self.n_iter)]

        for i, (downscale_weight, upscale_weight) in enumerate(
            zip(self.downscale_weights, self.upscale_weights)
        ):

            convertion = False
            if downscale_weight.data.dtype != torch.float:
                convertion = True

                downscale_type = downscale_weight.data.dtype
                downscale_device = downscale_weight.data.device
                downscale_weight = downscale_weight.to(torch.float)

                upscale_type = upscale_weight.data.dtype
                upscale_device = upscale_weight.data.device
                upscale_weight = upscale_weight.to(torch.float)

            down_q, down_r = torch.linalg.qr(downscale_weight)
            up_q, up_r = torch.linalg.qr(upscale_weight.t())

            chunk_size = int(scale * self.rank)

            # Split down_q, down_r into n_iter parts
            down_q_sizes = (chunk_size,) * (self.n_iter - 1) + (
                self.rank - (self.n_iter - 1) * chunk_size,
            )
            split_down_q = torch.split(down_q, down_q_sizes, dim=1)

            down_r_sizes = (chunk_size,) * (self.n_iter - 1) + (
                self.rank - (self.n_iter - 1) * chunk_size,
            )
            split_down_r: Tuple[torch.Tensor] = torch.split(down_r, down_r_sizes, dim=0)

            # print("Split down_q:", [x.shape for x in split_down_q])
            # print("Split down_r:", [x.shape for x in split_down_r])

            # Split up_q, up_r into n_iter parts
            up_q_sizes = (chunk_size,) * (self.n_iter - 1) + (
                self.rank - (self.n_iter - 1) * chunk_size,
            )
            split_up_q = torch.split(up_q, up_q_sizes, dim=1)

            up_r_sizes = (chunk_size,) * (self.n_iter - 1) + (
                self.rank - (self.n_iter - 1) * chunk_size,
            )
            split_up_r = torch.split(up_r, up_r_sizes, dim=0)

            # print("Split up_q:", [x.shape for x in split_up_q])
            # print("Split up_r:", [x.shape for x in split_up_r])

            # Mingling
            for j, (down_q, down_r, up_q, up_r) in enumerate(
                zip(split_down_q, split_down_r, split_up_q, split_up_r)
            ):
                down_weight = down_q  # torch.mm(down_q, down_r)
                up_weight = torch.mm(down_r, torch.mm(up_q, up_r).t())
                # print(down_weight.shape, up_weight.shape, torch.mm(up_q, up_r).t().shape, down_q.shape, down_r.shape, up_q.shape, up_r.shape)
                if convertion:
                    down_weight = down_weight.to(downscale_device).type(downscale_type)
                    up_weight = up_weight.to(upscale_device).type(upscale_type)
                # print(new_downscale_weights[j].shape, new_upscale_weights[j].shape)
                # new_downscale_weights[j] += down_weight
                # new_upscale_weights[j] += up_weight
                # print(i*chunk_size, (i+1)*chunk_size, down_weight.shape, up_weight.shape)

                cum_sum = [0] + list(torch.cumsum(torch.tensor(down_q_sizes), dim=0))
                # print(cum_sum, cum_sum[i],cum_sum[i+1])
                new_downscale_weights[i][:, cum_sum[j] : cum_sum[j + 1]] += down_weight
                new_upscale_weights[i][cum_sum[j] : cum_sum[j + 1], :] += up_weight

            # for j, (down_q1, down_r1, _, _) in enumerate(zip(split_down_q, split_down_r, split_up_q, split_up_r)):
            #     mix = torch.zeros(self.in_features, self.out_features).to(downscale_weight.device)
            #     for k, (_, _, up_q2, up_r2) in enumerate(zip(split_down_q, split_down_r, split_up_q, split_up_r)):
            #         mix_down = torch.mm(down_q1, down_r1)
            #         mix_up = torch.mm(up_q2, up_r2).t()
            #         mix += torch.mm(mix_down, mix_up)
            #         # down_weight = torch.mm(down_q1, down_r2)
            #         # up_weight = torch.mm(up_q2, up_r1).t()
            #     print(mix.shape, chunk_size)
            #     down_weight, up_weight = torch.linalg.qr(mix)
            #     down_weight = down_weight[:, :self.rank]
            #     up_weight = up_weight[:self.rank, :]
            #     if convertion:
            #         down_weight = down_weight.to(downscale_device).type(downscale_type)
            #         up_weight = up_weight.to(upscale_device).type(upscale_type)
            #     print(down_weight.shape, up_weight.shape)
            #     new_downscale_weights[k][j] += down_weight
            #     new_upscale_weights[k][j] += up_weight

        for i, (downscale_weight, upscale_weight) in enumerate(
            zip(new_downscale_weights, new_upscale_weights)
        ):
            self.downscale_weights[i].data = (
                downscale_weight  # torch.sum(downscale_weight, dim=0)
            )
            self.upscale_weights[i].data = (
                upscale_weight  # torch.sum(upscale_weight, dim=0)
            )

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
