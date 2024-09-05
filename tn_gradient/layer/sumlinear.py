import torch
import torch.nn as nn

from math import sqrt

class SumLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 16,
        n_iter: int = 5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SumLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter
        self.rank = rank

        self.downscale_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(in_features, rank, **factory_kwargs)) for _ in range(n_iter)]
        )
        self.upscale_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, out_features, **factory_kwargs)) for _ in range(n_iter)]
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:

        for downscale_weight, upscale_weight in zip(self.downscale_weights, self.upscale_weights):
            nn.init.kaiming_uniform_(downscale_weight, a=sqrt(5))
            nn.init.kaiming_uniform_(upscale_weight, a=sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.downscale_weights[0])
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for downscale_weight, upscale_weight in zip(self.downscale_weights, self.upscale_weights):
            downproj = torch.einsum("bi,ir->br", x, downscale_weight)
            if out is not None:
                out += torch.einsum("br,ro->bo", downproj, upscale_weight)
            else:
                out = torch.einsum("br,ro->bo", downproj, upscale_weight)

        if self.bias is not None:
            out += self.bias
        return out
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, n_iter={self.n_iter}"
