import os
import json
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import transformers
import wandb

from loguru import logger

def get_all_schedulers(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
    cycle_ratio=1.0,
    reset_after_warmup=False
):
    if not isinstance(num_training_steps, tuple):
        num_training_steps = (num_training_steps, ) * len(optimizer.param_groups)
        cycle_length = (cycle_length,) * len(optimizer.param_groups)
        cycle_ratio = (cycle_ratio,) * len(optimizer.param_groups)

    lambda_schedulers = []
    for i in range(len(num_training_steps)):
        scheduler = get_scheculer(
            optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps[i],
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            cycle_length=cycle_length[i],
            restart_warmup_steps=restart_warmup_steps,
            adjust_step=adjust_step,
            last_epoch=last_epoch,
            cycle_ratio=cycle_ratio[i],
            param_group_indices=[i],
            reset_after_warmup=reset_after_warmup
        )
        lambda_schedulers.append(scheduler)
    
    if len(lambda_schedulers) == 1:
        lambda_schedulers = lambda_schedulers[0]
    return LambdaLR(optimizer, lambda_schedulers, last_epoch)


def get_scheculer(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
    cycle_ratio=1.0,
    param_group_indices=None,
    reset_after_warmup=False
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    warmup_steps = int(warmup_steps * num_training_steps)

    if scheduler_type == "linear":
        # NO GROUP INDICES!
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            cycle_ratio=cycle_ratio,
            param_group_indices=param_group_indices,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
            cycle_ratio=cycle_ratio,
            param_group_indices=param_group_indices,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1, cycle_ratio=1.0, param_group_indices=None):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
        cycle_ratio=cycle_ratio
    )
    return lr_lambda


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return lr_lambda

@torch.no_grad()
def random_pruning(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor = tensor * random_pruning_mask
    return tensor


@torch.no_grad()
def magnitude_pruning(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor = tensor * mask.to(dtype=tensor.dtype)
    return tensor


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio, cycle_ratio=1.0):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    cycle_step = current_step % cycle_length
    cycle_number = current_step // cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps)) * (cycle_ratio ** cycle_number)

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps)) 
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay) * (cycle_ratio ** cycle_number)


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps < num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps < restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

def reset_scheduler(optimizer, group_id):
    # Clear the optimizer states of the specified param groups
    group = optimizer.param_groups[group_id]
    for param in group["params"]:
        state = optimizer.state[param]
        # Exponential moving average of gradient values
        state["exp_avg"] = torch.zeros_like(
            param, memory_format=torch.preserve_format
        )
        # Exponential moving average of squared gradient values
        state["exp_avg_sq"] = torch.zeros_like(
            param, memory_format=torch.preserve_format
        )
        if group["amsgrad"]:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state["max_exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )

        state["step"] = torch.zeros_like(state["step"])

