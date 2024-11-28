import torch
import torch.nn as nn

from opt_einsum import contract_expression, contract_path

from math import sqrt
from typing import List

from dataclasses import dataclass

from tn_gradient.utils import qr_weight, perturbe_random, randhaar, randuptri

@dataclass
class SoWArgs:
    device: str = None
    dtype: torch.dtype = None

    init_method: str = "normal_QR"

    rank: int = 16
    n_iter: int = 5
    scale: float = 1


class SoWParameter(nn.ParameterList):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_iter: int = 1,
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

    def from_weights(self, weights: List[torch.Tensor]) -> None:
        for i, weight in enumerate(weights):
            self[i].data = weight.data

    def extra_repr(self) -> str:
        return f"{self.n_iter} x ({self.in_features}, {self.out_features})"


class SoWLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 16,
        n_iter: int = 5,
        scale: float = 1,
        init_method: str = "normal_QR",
        device=None,
        dtype=None,
        init_params=True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SoWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter
        self.rank = rank
        self.scale = scale
        self.virtual_rank = min(rank * n_iter, in_features, out_features)

        self.acc_upweight = nn.Parameter(torch.empty(0), requires_grad=False)
        self.acc_downweight = nn.Parameter(torch.empty(0), requires_grad=False)

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

        if init_params:
            self.reset_parameters()

    def reset_parameters(self, reset_scale=1.0) -> None:
        if "normal_QR" in self.init_method:
            weight = torch.zeros((self.in_features, self.out_features)).to("cuda")

        for i in range(self.n_iter):
            if i / self.n_iter >= 1 - reset_scale:
                if self.init_method == "normal_QR":
                    nn.init.normal_(weight, mean=0.0, std=0.02) # hardcoded value std=0.02 from Llama config
                    q_weight, r_weight = qr_weight(weight, self.rank)
                    self.downscale_weights[i] = q_weight.to(self.downscale_weights[i].device).type(self.downscale_weights[i].dtype).contiguous()
                    self.upscale_weights[i] = r_weight.to(self.upscale_weights[i].device).type(self.upscale_weights[i].dtype).contiguous()
                else:
                    nn.init.normal_(self.downscale_weights[i], std=0.02)
                    nn.init.normal_(self.upscale_weights[i], std=0.02)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(0, device=self.downscale_weights[0].device)
        if self.acc_downweight.numel() != 0 and self.acc_upweight.numel() != 0:
            out = (x @ self.acc_downweight) @ self.acc_upweight
        elif self.acc_downweight.numel() != 0 and self.acc_upweight.numel() == 0:
            out = x @ self.acc_downweight

        for downscale_weight, upscale_weight in zip(
            self.downscale_weights, self.upscale_weights
        ):
            output = x @ downscale_weight @ upscale_weight
            if out.numel() != 0:
                out += output * self.scale
            else:
                out = output * self.scale

        if self.bias is not None:
            out += self.bias

        return out
    
    def accumulate(self):

        # Accumulate the weights
        accumalation = self.scale * torch.sum(torch.stack([
            a.detach() @ b.detach() 
            for a, b in zip(self.downscale_weights, self.upscale_weights)
        ]), dim=0).detach()

        # Compute the full W_acc matrix from the previous accumulated weights
        if self.acc_downweight.numel() != 0 and self.acc_upweight.numel() != 0:
            accumalation = accumalation.to(self.acc_upweight.device) + self.acc_downweight @ self.acc_upweight
        elif self.acc_downweight.numel() != 0 and self.acc_upweight.numel() == 0:
            accumalation = accumalation.to(self.acc_downweight.device) + self.acc_downweight

        # Perform QR decomposition to get the new accumulated weights
        # only if the virtual rank is less than the full-rankness
        if self.virtual_rank < min(self.in_features, self.out_features):

            Q, R = qr_weight(accumalation, rank=self.virtual_rank)
            self.acc_downweight = nn.Parameter(Q.contiguous(), requires_grad=False)
            self.acc_upweight = nn.Parameter(R.contiguous(), requires_grad=False)
            
            self.virtual_rank = min(self.virtual_rank + self.rank * self.n_iter, self.in_features, self.out_features)
        else:
            self.acc_downweight = nn.Parameter(accumalation, requires_grad=False)
            self.acc_upweight = nn.Parameter(torch.empty(0), requires_grad=False)

        torch.cuda.empty_cache()

        # Reset weights
        new_downscale_weights = [torch.zeros_like(x) for x in self.downscale_weights]
        new_upscale_weights = [torch.zeros_like(x) for x in self.upscale_weights]

        # Change initializations
        if "normal_QR" in self.init_method:
            weight = torch.zeros((self.in_features, self.out_features))
            weight = weight.to(self.acc_downweight.device)              
            weight = weight.type(self.acc_downweight.dtype)

        for i in range(0, self.n_iter):
            # Do not reinitialize the upscale weights other than zero for continuity of the accumulation
            if self.init_method == "normal_QR":
                nn.init.normal_(weight, mean=0.0, std=0.02) # hardcoded value std=0.02 from Llama/Roberta config
                q_weight, _ = qr_weight(weight, self.rank)
                new_downscale_weights[i] = q_weight.contiguous()
            else:
                nn.init.normal_(new_downscale_weights[i], std=0.02)


        self.downscale_weights.from_weights(new_downscale_weights)
        self.upscale_weights.from_weights(new_upscale_weights)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, n_iter={self.n_iter}"