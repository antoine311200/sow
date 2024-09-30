import torch
import torch.nn as nn

import math
from typing import Callable, Iterable, Tuple

from tn_gradient.tt import TensorTrain
from galore_torch.galore_projector import GaLoreProjector

class TTAdam(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        correct_bias: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            correct_bias=correct_bias
        )
        super(TTAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_shape = grad.shape
                if grad.is_sparse:
                    raise RuntimeError("TTAdam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "ranks" in group and False:
                    if state["step"] % 200 == 0:
                        data = grad.data.float()
                        Q, _ = torch.linalg.qr(data)
                        Q = Q[:, :128]
                        state["projector"] = Q
                        state["projector"] = state["projector"].to(grad.device).type(grad.dtype)
                    grad = torch.matmul(state["projector"].t(), grad)
                    grad_shape = grad.shape

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_expr"] = None
                elif "ranks" in group:
                    matrix = state["exp_avg"].to_matrix(grad_shape).to(grad.device).type(grad.dtype)
                    state["exp_avg_expr"] = state["exp_avg"].contract_expr
                    state["exp_avg"] = matrix

                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["exp_avg_sq_expr"] = None
                elif "ranks" in group:
                    matrix = state["exp_avg_sq"].to_matrix(grad_shape).to(grad.device).type(grad.dtype)
                    state["exp_avg_sq_expr"] = state["exp_avg_sq"].contract_expr

                    state["exp_avg_sq"] = matrix
                    state["exp_avg_sq"][state["exp_avg_sq"] < 0] = 0


                state["step"] += 1
                
                m, v = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                m.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = v.sqrt().add_(group["eps"])
                
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = m / denom

                if "ranks" in group and False:
                    norm_grad = torch.matmul(state["projector"], norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                if "ranks" in group:
                    state["exp_avg"] = TensorTrain.from_matrix(m, ranks=group["ranks"], padding=True)
                    state["exp_avg_sq"] = TensorTrain.from_matrix(v, ranks=group["ranks"], padding=True)

        return loss
    

class TTRAdam(torch.optim.Optimizer):
    pass
    # def __init__(
    #     self,
    #     params: Iterable[nn.parameter.Parameter],
    #     lr: float = 1e-3,
    #     betas: Tuple[float, float] = (0.9, 0.999),
    #     eps: float = 1e-8,
    #     weight_decay: float = 0,
    #     amsgrad: bool = False,
    #     correct_bias: bool = True,
    # ):
    #     defaults = dict(
    #         lr=lr,
    #         betas=betas,
    #         eps=eps,
    #         weight_decay=weight_decay,
    #         amsgrad=amsgrad,
    #         correct_bias=correct_bias
    #     )
    #     super(TTRAdam, self).__init__(params, defaults)

    # @torch.no_grad()
    # def step(self, closure: Callable = None):
    #     """
    #     Performs a single optimization step.

    #     Arguments:
    #         closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
    #     """
    #     loss = None
    #     if closure is not None:
    #         loss = closure()

    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #             grad = p.grad
    #             grad_shape = grad.shape
    #             if grad.is_sparse:
    #                 raise RuntimeError("TTAdam does not support sparse gradients, please consider SparseAdam instead")

    #             state = self.state[p]

    #             if "step" not in state:
    #                 state["step"] = 0

    #             if "ranks" in group:
    #                 d_p = TensorTrain.from_matrix(grad, ranks=group["ranks"], padding=True)
    #             else:
    #                 d_p = grad

    #             if "exp_avg" not in state:
    #                 if "ranks" in group:
    #                     state["exp_avg"] = TensorTrain.zeros(group["ranks"], d_p.input_shape, d_p.output_shape, device=d_p.device)
    #                 else:
    #                     state["exp_avg"] = torch.zeros_like(d_p, device=d_p.device)
    #             if "exp_avg_sq" not in state:
    #                 state["exp_avg_sq"] = torch.scalar_tensor(0.0, device=d_p.device)

    #             state["step"] += 1
                
    #             m, v = state["exp_avg"], state["exp_avg_sq"]
    #             beta1, beta2 = group["betas"]

    #             m = beta1 * m + (1 - beta1) * d_p
    #             v = beta2 * v + (1 - beta2) * d_p.norm()
    #             denom = v.sqrt().add_(group["eps"])

    #             step_size = group["lr"]
    #             if group["correct_bias"]:  # No bias correction for Bert
    #                 bias_correction1 = 1.0 - beta1 ** state["step"]
    #                 bias_correction2 = 1.0 - beta2 ** state["step"]
    #                 step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

    #             # compute norm gradient
    #             norm_grad = (1 / denom) * m
    #             if "ranks" in group:
    #                 norm_grad = norm_grad.round(d_p.input_shape[0] * d_p.output_shape[0])
    #                 norm_grad = norm_grad.to_matrix(grad_shape)

    #             p.add_(norm_grad, alpha=-step_size)
                
    #             if group["weight_decay"] > 0.0:
    #                 p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

    #     return loss