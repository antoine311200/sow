import torch
import torch.nn as nn

import math
from typing import Callable, Iterable, Tuple

from tn_gradient.tt import TensorTrain

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

                # if "ranks" in group:
                #     # d_p = TensorTrain.from_matrix(grad, ranks=group["ranks"], padding=True)
                # else:

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                elif "ranks" in group:
                    # print(state["exp_avg"].input_shape, state["exp_avg"].output_shape)
                    # print(grad_shape)
                    # print(state["exp_avg"].cores[1][:, 2, :])
                    # state["exp_avg"] = state["exp_avg"].to_matrix(grad_shape).requires_grad_(False)
                    # print(state["exp_avg"][0, 0])
                    # Print max value of state["exp_avg"]
                    # print("m", torch.max(state["exp_avg"].abs()))
                    # state["exp_avg"] = TensorTrain.from_matrix(state["exp_avg"], ranks=group["ranks"], padding=True)
                    # # print(max([core.abs().max() for core in state["exp_avg"].cores]))
                    state["exp_avg"] = state["exp_avg"].to_matrix(grad_shape)

                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                elif "ranks" in group:
                    # print("v", torch.max(state["exp_avg_sq"].abs()))
                    # state["exp_avg_sq"] = TensorTrain.from_matrix(state["exp_avg_sq"], ranks=group["ranks"], padding=True)
                    # print(max([core.abs().max() for core in state["exp_avg_sq"].cores]))
                    state["exp_avg_sq"] = state["exp_avg_sq"].to_matrix(grad_shape)
                    state["exp_avg_sq"][state["exp_avg_sq"] < 0] = 0

                # if "correct_bias" not in group:
                #     group["correct_bias"] = True

                # if "exp_avg" not in state:
                #     if "ranks" in group:
                #         state["exp_avg"] = TensorTrain.zeros(group["ranks"], d_p.input_shape, d_p.output_shape, device=d_p.device)
                #     else:
                #         state["exp_avg"] = torch.zeros_like(d_p)
                # if "exp_avg_sq" not in state:
                #     if "ranks" in group:
                #         state["exp_avg_sq"] = TensorTrain.zeros(group["ranks"], d_p.input_shape, d_p.output_shape, device=d_p.device)
                #     else:
                #         state["exp_avg_sq"] = torch.zeros_like(d_p)

                state["step"] += 1
                
                m, v = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # m = beta1 * m + (1 - beta1) * d_p
                # v = beta2 * v + (1 - beta2) * (d_p * d_p)
                m.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = v.sqrt().add_(group["eps"])
                

                # if "ranks" in group:
                #     # v = v.round(d_p.input_shape[0] * d_p.output_shape[0])

                #     m_ = m.to_matrix(grad_shape)
                #     v_ = v.to_matrix(grad_shape)

                #     # During reconstruction to matrix some values that should be 0 can
                #     # become negative, so we need to correct that
                #     v_[v_ < 0] = 0
                # else:
                # m_ = m
                # v_ = v

                # m_hat = (1 / (1 - beta1)) * m
                # v_hat = (1 / (1 - beta2)) * v

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = m / denom

                p.add_(norm_grad, alpha=-step_size)
                
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                if "ranks" in group:
                    state["exp_avg"] = TensorTrain.from_matrix(m, ranks=group["ranks"], padding=True)
                    state["exp_avg_sq"] = TensorTrain.from_matrix(v, ranks=group["ranks"], padding=True)
                    
        return loss