import torch
import torch.nn as nn

from typing import Callable, Iterable, Tuple

from tn_gradient.tt import TensorTrain

class TTSGD(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super(TTSGD, self).__init__(params, defaults)

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

        # count = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_shape = grad.shape
                if grad.is_sparse:
                    raise RuntimeError("TTSGD does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "ranks" in group:
                    d_p = TensorTrain.from_matrix(grad, ranks=group["ranks"], padding=True)
                else:
                    d_p = grad

                if group["weight_decay"] != 0:
                    d_p = d_p.add(p, alpha=group["weight_decay"])

                if group["momentum"] != 0:
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = d_p.clone().detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf = group["momentum"] * buf + (1 - group["dampening"]) * d_p
                    if group["nesterov"]:
                        d_p = d_p + group["momentum"] * buf
                    else:
                        d_p = buf

                if "ranks" in group:
                    d_p = d_p.to_matrix(grad_shape)

                p.add_(-group["lr"] * d_p)
                
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # count += 1

        # print(f"Performed {count} optimization steps")
        return loss
    