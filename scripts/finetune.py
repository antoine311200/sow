import os
import sys
from typing import List, Dict, Any

import fire
import torch
from torch import nn
from torch.optim import AdamW
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
# sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    # BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel, get_scheduler  # noqa: F402

import wandb

from tn_gradient.layer.sow import SoWLinear
from tn_gradient.prepare import prepare_sow, SoWConfig
from utils.training_utils import reset_optimizer

from torchcolor.printer import Printer
from torchcolor.strategy import TrainableStrategy

class SoWTrainer(transformers.Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        if getattr(self, "freq_step", None):
            self.freq_step += 1
        else:
            self.freq_step = 1

        if (
            self.state.global_step > 0 and
            self.state.global_step % self.args.accumulation_steps == 0 and
            self.freq_step % self.args.accumulation_steps == 0 and 
            self.freq_step > 0
        ):
            print(f"Accumulation, Scaling & Reset optimizer states (step {self.state.global_step})")

            scaling = 1/self.args.rank
            for _, module in model.named_modules():
                if isinstance(module, SoWLinear):
                    module.accumulate()
                    module.scale = scaling
            reset_optimizer(self.optimizer, group_id=2) # reset the second parameter group
            self.freq_step = 1

        wandb.log({
            "train_loss": loss.detach().item(),
            "step": self.state.global_step,
        })

        return loss.detach() / self.args.gradient_accumulation_steps

from dataclasses import dataclass, field

@dataclass
class SoWTrainingArguments(transformers.TrainingArguments):
    rank: int = 10,
    accumulation_steps: int = 1000,
    sow_lr: float = 2e-4,
    target_modules: list = field(default_factory=list)

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./sow-alpaca",
        adapter_name: str = "sow",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 400000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # SoW hyperparams
        rank: int = 10,
        accumulation_steps: int = 1000,
        sow_lr: float = 2e-4,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"rank: {rank}\n",
        f"accumulation_steps: {accumulation_steps}\n",
        f"sow_lr: {sow_lr}\n",
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # os.environ["WANDB_CAPTURE_OUTPUT"] = "true"
    # os.environ["WANDB_WATCH"] = "all"

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    # elif adapter_name == "bottleneck":
    #     config = BottleneckConfig(
    #         bottleneck_size=bottleneck_size,
    #         non_linearity=non_linearity,
    #         adapter_dropout=adapter_dropout,
    #         use_parallel_adapter=use_parallel_adapter,
    #         use_adapterp=use_adapterp,
    #         target_modules=target_modules,
    #         scaling=scaling,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "sow":
        if "llama" in base_model.lower():
            target_modules = [
                "q_proj", "k_proj", "v_proj",
                # "o_proj", "gate_proj",
                "up_proj", "down_proj"
            ]
        elif "roberta" in base_model:
            target_modules = ["query", "key", "value", "output.dense", "intermediate.dense"]
        print("Preparing SoW model")

        sow_config = SoWConfig(
            target_modules=target_modules,
            rank=rank,
            device="cuda"
        )
        model = prepare_sow(model, sow_config)
        print("SoW model prepared")
        
        printer = Printer(TrainableStrategy())
        printer.print(model, display_legend=True)

    if adapter_name != "sow":
        model = get_peft_model(model, config)
    
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_step if val_set_size > 0 else None,
        save_steps=save_step,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )
    SpecificTrainer = transformers.Trainer
    
    if adapter_name == "sow":
        no_decay = ["bias", "LayerNorm.weight"]
        special_params = []
        for module_name, module in model.named_modules():
            if not isinstance(module, SoWLinear):
                continue

            if not any(target_key in module_name for target_key in target_modules):
                continue

            for downscale_weight in module.downscale_weights:
                if downscale_weight.requires_grad:
                    special_params.append(downscale_weight)
            for upscale_weight in module.upscale_weights:
                if upscale_weight.requires_grad:
                    special_params.append(upscale_weight)

        id_params = [id(p) for p in special_params]
        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in id_params]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in trainable_params if not any(nd in n for nd in no_decay)],
                'lr': learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in trainable_params if any(nd in n for nd in no_decay)],
                'lr': learning_rate,
                "weight_decay": 0.0,
            },
            { 'params': special_params, 'lr': sow_lr, 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

        SpecificTrainer = SoWTrainer
        
        wandb_run_name = f"sow_r_{rank}_freq_{accumulation_steps}_lr_{learning_rate}_slr_{sow_lr}"
        training_args = SoWTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            fp16=False,
            logging_steps=100,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            rank=rank,
            accumulation_steps=accumulation_steps,
            sow_lr=sow_lr,
            target_modules=target_modules,
        )

    from math import ceil
    
    num_update_steps_per_epoch = len(train_data) // training_args.gradient_accumulation_steps
    max_steps = ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = training_args.get_warmup_steps(max_steps)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )

    trainer = SpecificTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        optimizers=(optimizer, lr_scheduler)
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    torch.save(training_args, os.path.join(output_dir, "training_args.bin"))

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
