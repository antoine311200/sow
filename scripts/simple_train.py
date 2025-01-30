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
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd.profiler import record_function

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_model, load_file

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from tn_gradient.optimizer.ttsgd import TTSGD
from tn_gradient.tt import TensorTrain
from tn_gradient.layer.sow import SoWLinear
from tn_gradient.prepare import prepare_sow, accumulate, load_sow, SoWConfig

from utils.dataloader import PreprocessedIterableDataset, batch_fn
from utils.args_utils import check_args_torchrun_main
from utils.training_utils import get_all_schedulers, reset_optimizer
from utils.memory_utils import calculate_optimizer_memory_usage, calculate_weight_usage

from tn_gradient.utils import __colorized_str__
torch.nn.Module.__str__ = __colorized_str__

from galore_torch import GaLoreAdamW, GaLoreAdafactor

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
    parser.add_argument("--beta_1", type=float, default="0")
    parser.add_argument("--beta_2", type=float, default="0")

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


def save_model(model, optimizer, scheduler, training_state_checkpoint, run_config):
    update_step = training_state_checkpoint["update_step"]
    global_step = training_state_checkpoint["global_step"]

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

    with open(f"{current_model_directory}/training_state.json", "w") as f:
        json.dump(training_state_checkpoint, f, indent=4)
        
    # save wandb related info
    wandb_info = {
        "wandb_id": wandb.run.id,
    }
    with open(os.path.join(args.save_dir, "wandb.json"), "w") as f:
        json.dump(wandb_info, f, indent=4)

    with open(os.path.join(args.save_dir, "training_config.yaml"), "w") as f:
        yaml.dump(vars(args), f)












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

    
    wandb_id = None
    if args.continue_from is not None:
        logger.info("*" * 40)

        base_folder = args.continue_from
        resume_folder = args.continue_from
        if not os.path.basename(os.path.normpath(base_folder)).startswith("model_"):
            model_dirs = [d for d in os.listdir(base_folder) if d.startswith(f"model_")]
            if len(model_dirs) == 0:
                logger.warning(f"Save directory {base_folder} exists, but does not contain any models.")
                logger.warning("Starting training from scratch.")

            model_dirs = sorted(model_dirs, key=lambda x: int(x.split("_")[-1]))
            resume_folder = os.path.join(base_folder, model_dirs[-1])
        else:
            base_folder = os.path.dirname(resume_folder)

        logger.info(f"Restarting training from {args.continue_from}")
        with open(os.path.join(base_folder, "wandb.json")) as f:
            wandb_id = json.load(f)["wandb_id"]
        logger.info(f"Resuming training from {resume_folder} with wandb id {wandb_id}")

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
            id=wandb_id,
            resume="allow",
            mode="disabled" if args.wandb_off or args.eval else None
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
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        sow_config = SoWConfig(
            target_modules=target_modules,
            rank=args.rank,
            init_method=args.init_method,
            scale=args.sow_scale,
            decompose=None,
            device="cuda"
        )


        logger.info("Preparing SoW model")
        model = prepare_sow(model, sow_config)
        # model = prepare_sow(model, target_modules, decompose=False, args=sow_args)
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
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "model.safetensors")

        if args.architecture == "sow":
            # Need to check wether we load a non-SoW pretrained model or a statedict of a SoW model 
            load_sow(model, checkpoint_path)
        else:
            load_model(model, checkpoint_path)
        
        
        logger.info(f"Model successfully loaded (strict=True policy)")
        print(model.model.layers[1].self_attn.q_proj.downscale_weights[0].data)

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
        logger.info("Getting LoRA parameters")

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
        logger.info("Getting GaLore parameters")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        for module_name, module in model.named_modules():

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

    memory_usage, memory_usage_sow, memory_usage_accum, memory_buffer = calculate_weight_usage(model)
    memory_train_usage = sum(p.nelement() * p.element_size() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n{model}\n")
    logger.info(f"Total params (start): {memory_usage / (1024 * 1024):.2f}MiB [{memory_usage / 1e6}Mb]")
    logger.info(f"Total params (end): {(memory_usage + memory_usage_accum) / (1024 * 1024):.2f}MiB (+{memory_usage_accum/ (1024 * 1024):.2f}MiB)")
    logger.info(f"Trainable params: {memory_train_usage / (1024 * 1024):.2f}MiB [{memory_train_usage / 1e6}Mb]")
    logger.info(f"Buffer params: {memory_buffer / (1024 * 1024):.2f}MiB")
    # logger.info(f": {(memory_usage + memory_usage_accum + memory_train_usage) / (1024 * 1024):.2f}MiB")

    if args.architecture == "sow":
        logger.info(f"SoW params: {memory_usage_sow / (1024 * 1024):.2f}MiB")

    if args.optimizer.lower() == "galore_adamw":
        logger.info(f"Trainable params (GaLoRe): {len(trainable_params)}")
        logger.info(f"Special params (GaLoRe): {len(special_params)}")
        optimizer = GaLoreAdamW([
            {'params': trainable_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {
                'params': special_params, 
                'lr': args.sow_lr, 'weight_decay': args.weight_decay, 
                'rank': args.galore_rank, 'update_proj_gap': args.update_proj_gap,
                'scale': args.galore_scale, 'proj_type': args.proj_type
            }
        ])
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            [
                {'params': trainable_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                {
                    'params': special_params, 
                    'lr': args.sow_lr, 'weight_decay': args.weight_decay, 
                    'rank': args.galore_rank, 'update_proj_gap': args.update_proj_gap,
                    'scale': args.galore_scale, 'proj_type': args.proj_type
                }
            ],
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
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
        logger.info("Setting scheduler to the same state as in the checkpoint")
        for _ in range(update_step):
            scheduler.step()
        logger.info(f"Scheduler state restored from {resume_folder}")
        logger.info(f"Current lr is {optimizer.param_groups[0]['lr']}")
        if args.architecture == "sow":
            logger.info(f"Current SoW lr is {optimizer.param_groups[1]['lr']}")

        optimizer_path = os.path.join(resume_folder, "optimizer.pt")
        optimizer_checkpoint = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        
        update_step = optimizer_checkpoint["update_step"]
        global_step = optimizer_checkpoint["global_step"]

        logger.info("Optimizer & Scheduler state dict loaded")

        _training_config_path = os.path.join(resume_folder, "training_config.yaml")
        if os.path.exists(_training_config_path):
            with open(_training_config_path) as f:
                _old_training_config = yaml.safe_load(f)
            if args.batch_size != _old_training_config["batch_size"]:
                raise RuntimeError("Cannot resume from a checkpoint with a different batch size.")
            if args.eval:
                args = _old_training_config


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
        logger.info(f"Eval loss : {total_loss} | Perplexity : {np.exp(total_loss)}")

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

        # print(global_step % args.gradient_accumulation, update_step, offset, (update_step - offset) % accumulation_step)

        if (global_step % args.gradient_accumulation or args.gradient_accumulation == 1) and update_step > offset and (update_step - offset) % accumulation_step == 0 and args.architecture == "sow":
            logger.info(f"\nAccumulation & Reset optimizer states (step global: {global_step} - local: {update_step})")
            accumulate(model)
            reset_optimizer(optimizer, group_id=1) # reset the second parameter group    

        if global_step % args.gradient_accumulation != 0:
            continue

        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        if update_step == 10: # 50
            
            # Get the optimizer memory usage
            optimizer_memory_usage, _ = calculate_optimizer_memory_usage(optimizer)
            full_optimizer_memory_usage = optimizer_memory_usage
            full_optimizer_memory_usage = full_optimizer_memory_usage / (1024 * 1024)
            
            logger.info(f"Optimizer memory usage: {full_optimizer_memory_usage:.2f} MiB")

            # import sys
            # sys.exit()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        update_step += 1
        update_time = time.time() - update_time
        
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            save_model(model, optimizer, scheduler, training_state_checkpoint, run_config)

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
            logger.info(f"Eval loss at step {update_step}: {total_loss} | Perplexity : {np.exp(total_loss)}")

        lr = optimizer.param_groups[0]["lr"]
        sow_lr = optimizer.param_groups[1]["lr"] if args.architecture == "sow" else None

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


if __name__ == "__main__":
    # print("Starting script")
    args = parse_args(None)
    main(args)

    # profiler = cProfile.Profile()

    # profiler.enable()
    # profiler.disable()

    # from io import StringIO
    # import pstats

    # s = StringIO()
    # ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    # X = 30
    # ps.print_stats(X)
    # print(s.getvalue())

    # profiler.print_stats(sort='cumtime')
