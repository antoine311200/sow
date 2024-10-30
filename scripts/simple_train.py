import os
import sys
import math
import time
import json
import random
import argparse
from typing import OrderedDict
import numpy as np
from functools import partial
import atexit
import cProfile
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd.profiler import record_function

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_model

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from tn_gradient.optimizer.ttsgd import TTSGD
# from tn_gradient.optimizer.ttadam import TTAdam, TTRAdam
from tn_gradient.tt import TensorTrain
# from tn_gradient.layer.tensor_linear import TensorTrainLinear
from tn_gradient.layer.sow import SoWLinear, SoWArgs
from tn_gradient.prepare import prepare_sow, accumulate

from galore_torch import GaLoreAdamW

from contextlib import contextmanager

@contextmanager
def conditional_with(condition: bool, arg):
    if condition:
        if arg:
            with arg as x:
                yield x
        else:
            yield
    else:
        yield

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_off", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--warmup_steps", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    parser.add_argument("--num_training_steps", type=int, default=1_000_000)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--monitor_memory", type=bool, default=False)
    parser.add_argument("--eval", default=False, action="store_true")

    # Sum Of (LO) Weight parameters
    parser.add_argument("--architecture", type=str, default="linear")
    parser.add_argument("--sow_accumulation", type=int, default=200)
    parser.add_argument("--sow_lr", type=float, default=0.0015)
    parser.add_argument("--sow_scale", type=float, default=1)
    parser.add_argument("--init_method", type=str, default="normal_QR")
    # parser.add_argument("--sow_init", type=str, default="qr_xe")
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--lr_decay", type=float, default="1.0")
    parser.add_argument("--reset_scheduler", default=False, action="store_true")
    parser.add_argument("--accumulate_after_warmup", default=False, action="store_true")

    # Galore parameters
    parser.add_argument("--galore_rank", type=int, default="128")
    parser.add_argument("--update_proj_gap", type=int, default="200")
    parser.add_argument("--galore_scale", type=float, default="0.25")
    parser.add_argument("--proj_type", type=str, default="std")

    parser.add_argument("--single_gpu", default=False, action="store_true")

    args = parser.parse_args(args)

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    return args






@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True, save_infos=True) #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 5_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens













def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)    

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.gradient_accumulation is None:
        args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
    if args.gradient_accumulation == 0:
        args.gradient_accumulation = 1
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")

    if args.architecture == "sow":
        run_name = f"{args.model_config.split('/')[-1].rstrip('.json')}-{args.optimizer}-{args.architecture}-lr-{args.sow_lr}-n-{args.n_iter}-r-{args.rank}-{args.sow_accumulation}"
    elif args.architecture == "sttlinear":
        run_name = f"{args.model_config.split('/')[-1].rstrip('.json')}-{args.optimizer}-{args.architecture}-n-{args.n_iter}-r-{args.rank}-o-{args.order}-i-{args.inner_rank}"
    else:
        run_name = f"{args.model_config.split('/')[-1].rstrip('.json')}-{args.optimizer}-{args.architecture}-lr-{args.lr}"

    if global_rank != 0: logger.remove()
    if global_rank == 0:
        wandb.init(
            project="sow-llama",
            name=run_name,
            mode="disabled" if args.wandb_off or args.eval else None
            # id=None if not args.continue_from else args.wandb_id,
            # resume=None if not args.continue_from else "must",
            # resume_from=args.wandb_id+'?_step=40000'
        )
    
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 42

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    model = AutoModelForCausalLM.from_config(model_config)
    
    if args.architecture == "sow" or args.architecture == "lora":
        sow_args = SoWArgs(
            rank=args.rank,
            n_iter=args.n_iter,
            init_method=args.init_method,
            scale=args.sow_scale,
            dtype=args.dtype,
            device=device,
        )

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        logger.info("Preparing SoW model")
        model = prepare_sow(model, target_modules, decompose=False, args=sow_args)
        logger.info("SoW model prepared")

        if args.architecture == "lora":
            from math import sqrt
            # Set accumulation step to be greater than the training steps
            args.sow_accumulation = args.num_training_steps + 1
            # Set accumulation matrix to be random and A to be zero
            for _, module in model.named_modules():
                if isinstance(module, SoWLinear):
                    module.acc_downweight = torch.zeros(
                        (module.in_features, module.out_features),
                        dtype=torch.bfloat16 if args.dtype in ["bf16", "bfloat16"] else None,
                        device=module.upscale_weights[0].device
                    )
                    nn.init.kaiming_uniform_(module.acc_downweight, a=sqrt(5))
                    for i in range(0, module.n_iter):
                        nn.init.zeros_(module.upscale_weights[i])
    
    local_step = 0 
    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        # checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        checkpoint_path = os.path.join(args.continue_from, "model.safetensors")
        # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        load_model(model, checkpoint_path)
        logger.info(f"Model successfully loaded (strict=True policy)")
        logger.info(model)

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    special_params = []
    if args.architecture == "sow" or args.architecture == "lora":

        for module_name, module in model.named_modules():
            if not isinstance(module, SoWLinear):
                continue

            if not any(target_key in module_name for target_key in target_modules):
                continue
            
            for downscale_weight in module.downscale_weights:
                special_params.append(downscale_weight)
            for upscale_weight in module.upscale_weights:
                special_params.append(upscale_weight)
        id_params = [id(p) for p in special_params]
        trainable_params = [p for p in model.parameters() if p.requires_grad and id(p) not in id_params]
    elif args.architecture == "galore":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        for module_name, module in model.named_modules():
            if not isinstance(module, SoWLinear):
                continue

            if not any(target_key in module_name for target_key in target_modules):
                continue

            special_params.append(module.weight)
            
        id_params = [id(p) for p in special_params]
        trainable_params = [p for p in model.parameters() if p.requires_grad and id(p) not in id_params]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })


    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    memory_usage, memory_usage_sow, memory_usage_accum = calculate_weight_usage(model)
    memory_train_usage = sum(p.nelement() * p.element_size() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n{model}\n")
    logger.info(f"Total params (start): {memory_usage / (1024 * 1024):.2f}MiB")
    logger.info(f"Total params (end): {(memory_usage + memory_usage_accum) / (1024 * 1024):.2f}MiB")
    logger.info(f"Trainable params: {memory_train_usage / (1024 * 1024):.2f}MiB")
    if args.architecture == "sow":
        logger.info(f"SoW params: {memory_usage_sow / (1024 * 1024):.2f}MiB")


    if args.optimizer.lower() == "SGD".lower():
        optimizer = TTSGD(
            trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer.lower() == "galore_adamw":
        optimizer = GaLoreAdamW([
            {'params': trainable_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {
                'params': special_params, 
                'lr': args.sow_lr, 'weight_decay': args.weight_decay, 
                'rank': args.galore_rank, 'update_proj_gap': args.update_proj_gap,
                'scale': args.galore_scale, 'proj_type': args.proj_type
            }
        ])
    elif args.optimizer.lower() == "AdamW".lower():
        optimizer = torch.optim.AdamW([
            {'params': trainable_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': special_params, 'lr': args.sow_lr, 'weight_decay': args.weight_decay}
        ])
    elif args.optimizer.lower() == "Adam".lower():
        optimizer = torch.optim.Adam([
            {'params': trainable_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': special_params, 'lr': args.sow_lr, 'weight_decay': args.weight_decay}
        ])
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")


    if args.architecture == "sow" and args.reset_scheduler:
        num_training_steps = (args.num_training_steps, args.sow_accumulation * args.gradient_accumulation)
        cycle_length = (None, args.sow_accumulation * args.gradient_accumulation)
        cycle_ratio = (1.0, args.lr_decay)
    else:
        num_training_steps = args.num_training_steps
        cycle_length = None
        cycle_ratio = 1.0
    
    scheduler = get_all_schedulers(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        restart_warmup_steps=args.warmup_steps,
        cycle_length=cycle_length,
        cycle_ratio=cycle_ratio,
        reset_after_warmup=args.accumulate_after_warmup
    )

    if args.continue_from:
        optimizer_path = os.path.join(args.continue_from, "optimizer.pt")
        optimizer_checkpoint = torch.load(optimizer_path)

        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        logger.info("Optimizer & Scheduler state dict loaded")

    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    pad_idx = tokenizer.pad_token_id
    update_time = time.time()

    if args.eval:
        model.eval()
        logger.info(f"Performing evaluation")
        total_loss, evaluated_on_tokens = evaluate_model(
            model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
        )
        logger.info(f"Eval loss : {total_loss}")

        sys.exit()

    ############## Training loop ##############

    profiler = None
    if args.monitor_memory:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
        logger.info("Starting memory profiling")

    atexit.register(cleanup, profiler, run_name)

    for _, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1
        
        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        # Need to accumulate at steps offset + k*accumulation_step for all k
        # Hence when update_step - offset if a multiple of accumulation_step

        offset = (int(args.warmup_steps * args.num_training_steps) if args.accumulate_after_warmup else 0)
        accumulation_step = int(args.gradient_accumulation * args.sow_accumulation)

        if update_step > offset and (update_step - offset) % accumulation_step == 0 and args.architecture == "sow":
            # print("\n")
            logger.info(f"Accumulation & Reset optimizer states (step global: {global_step} - local: {update_step})")
            accumulate(model)

            #  and args.reset_scheduler:

            # Clear the optimizer states of the second param groups
            group = optimizer.param_groups[1]
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

        if global_step % args.gradient_accumulation != 0:
            continue

        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        if update_step == 50:
            
            # Get the optimizer memory usage
            optimizer_memory_usage, _ = calculate_optimizer_memory_usage(optimizer)
            full_optimizer_memory_usage = optimizer_memory_usage
            full_optimizer_memory_usage = full_optimizer_memory_usage / (1024 * 1024)
            
            print("\n")
            logger.info(f"Optimizer memory usage: {full_optimizer_memory_usage:.2f} MiB")

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        update_step += 1
        update_time = time.time() - update_time



        
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            # Check if DistributedDataParallel is used
            if args.single_gpu:
                model.save_pretrained(current_model_directory, max_shard_size='100GB')
            else:
                model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
                
            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        lr = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1:
            sow_lr = optimizer.param_groups[1]["lr"]
        else:
            sow_lr = None
        
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "sow_lr": sow_lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                },
                step=global_step,
            )
        update_time = time.time()

def cleanup(profiler, name):
    logger.info("Cleaning up")
    if args.monitor_memory:
        try:
            torch.cuda.memory._dump_snapshot(f"{name}.pickle")
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot {e}")

        # Stop recording memory snapshot history.
        logger.info("Stopping memory profiling")
        
        torch.cuda.memory._record_memory_history(enabled=None)

import os
import math
import itertools
from functools import partial
from datetime import datetime

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, get_worker_info
import transformers

from loguru import logger

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def check_args_torchrun_main(args):

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    return args














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

def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)

def calculate_optimizer_memory_usage(optimizer):
    memory_usage = 0
    tt_memory_usage = 0
    for state in optimizer.state.values():
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor):
                memory_usage += tensor.nelement() * tensor.element_size()
            elif isinstance(tensor, TensorTrain):
                for core in tensor.cores:
                    tt_memory_usage += core.nelement() * core.element_size()
    return memory_usage, tt_memory_usage

def calculate_model_memory_usage(model):
    memory_usage = 0
    for param in model.parameters():
        if isinstance(param, torch.Tensor):
            memory_usage += param.nelement() * param.element_size()
    return memory_usage

def calculate_batch_memory_usage(batch, labels):
    memory_usage = 0
    for tensor in batch.values():
        memory_usage += tensor.nelement() * tensor.element_size()
    memory_usage += labels.nelement() * labels.element_size()
    return memory_usage

def calculate_weight_usage(model):
    memory_usage_sow = 0
    memory_usage_accum = 0
    memory_usage = sum(p.numel() for p in model.parameters())
    
    for param in model.parameters():
        if isinstance(param, torch.Tensor):
            memory_usage += param.nelement() * param.element_size()

    for _, module in model.named_modules():
        if isinstance(module, SoWLinear):
            for weight in module.upscale_weights:
                memory_usage_sow += weight.nelement() * weight.element_size()
            for weight in module.downscale_weights:
                memory_usage_sow += weight.nelement() * weight.element_size()
            
            memory_usage_accum += module.in_features * module.out_features
    
    return memory_usage, memory_usage_sow, memory_usage_accum

if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)

    profiler = cProfile.Profile()

    profiler.enable()
    main(args)
    profiler.disable()

    from io import StringIO
    import pstats

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    X = 30
    ps.print_stats(X)
    print(s.getvalue())

    # profiler.print_stats(sort='cumtime')
