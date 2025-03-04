import os
from math import sqrt
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from tqdm import tqdm
from tn_gradient.layer.sow import SoWLinear
from tn_gradient.utils import svd_weight

from peft import PeftModel, PeftConfig


# @dataclass
# class SoWArgs:
#     device: str = None
#     dtype: torch.dtype = None

#     init_method: str = "normal_QR"

#     rank: int = 16
#     n_iter: int = 5
#     scale: float = 1

class SoWConfig(PeftConfig):
    def __init__(self, target_modules, rank=16, scale=1.0, device="cpu", init_method="normal_QR", decompose="keep", **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.scale = scale
        self.target_modules = target_modules
        self.device = device

        self.init_method = init_method
        self.decompose = decompose

        self.peft_type = "LORA"


def prepare_sow(
    model, config: SoWConfig
    # target_modules, decompose: bool = 'qr', args: SoWArgs = SoWArgs()
):
    """Code for preparing the model for the Sum-of-Weights decomposition.

    Given a model and a list of target modules, this function will replace the target modules with the Sum-of-Weights decomposed layers.
    For pre-traning, the decomposition is not necessary as we can reduce the memory overhead with an empty accumulated weights until the
    first accumulation step. This is done by setting the `decompose` argument to False.
    For fine-tuning, the decomposition is necessary to keep the original weights of the model.

    This decomposition is done as follows:
    1. The weight matrix of the target module is decomposed into two matrices, Q and R, using the QR decomposition.
    2. The Q matrix is split into two matrices: the major part and the minor part.
        i. The major part accounts for the first Q.shape[0] - n_iter*rank columns of Q, that is the more information preserving part.
        ii. The minor part accounts for the last n_iter*rank columns of Q, that is the less information preserving part.
    3. The R matrix is split into two matrices: the major part and the minor part the same way as the Q matrix on its rows.
    4. The major part of Q and R are used to form the frozen accumulated weight matrix of the new layer.
    5. The minor part of Q and R are used to form the downscale and upscale weight matrices of the new layer by splitting them into rank-sized chunks.

    Args:
        model (nn.Module): The model to prepare.
        target_modules (list[str]): The list of target modules to replace.
        decompose (bool, optional): Whether to decompose the weights of the target modules. Defaults to True.
        args (SoWArgs, optional): The arguments for the Sum-of-Weights decomposition. Defaults to SoWArgs().

    Returns:
        nn.Module: The prepared model.
    """
    layers_to_replace = {}

    max_split = max([len(name.split(".")) for name in config.target_modules])

    def check_module(name, module):
        split_name = name.split(".")
        if isinstance(module, nn.Linear):
            if len(split_name) == 1 and split_name[0] in config.target_modules:
                return True

            for i in range(1, min(max_split + 1, len(split_name))):
                if ".".join(split_name[-i:]) in config.target_modules:
                    return True
        return False

    length = 0
    for name, module in model.named_modules():
        if check_module(name, module):
            # layers_to_replace[name] = module
            length += 1

    pbar = tqdm(total=length, desc="Prepare model", ncols=50)

    # for name, module in layers_to_replace.items():
    
    for name, module in model.named_modules():
        if check_module(name, module):
            # # Convertion to float is necessary for the QR decomposition
            # # as CUDA does not support QR decomposition for half precision
            convertion = False
            if module.weight.data.dtype != torch.float and config.decompose == "qr":
                convertion = True

                weight_type = module.weight.data.dtype
                weight_device = module.weight.data.device
                # module.weight = nn.Parameter(module.weight.to(torch.float), requires_grad=False) 

            # Create a blank Sum-of-Weights layer
            new_layer = SoWLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=config.rank,
                n_iter=1,
                scale=config.scale,
                init_method=config.init_method,
                bias=module.bias is not None,
                dtype=module.weight.data.dtype,
                device=config.device,
                init_params=config.decompose!='qr'
            )
            new_layer.virtual_rank = min(module.in_features, module.out_features)

            if config.decompose == 'qr':
                keep_rank = config.rank * 1
                Q, R = torch.linalg.qr(module.weight.data.T.to("cuda"))
                Q_major, Q_minor = (
                    Q[:, :-keep_rank],
                    Q[:, -keep_rank:],
                )
                R_major, R_minor = (
                    R[:-keep_rank, :],
                    R[-keep_rank:, :],
                )

                W = Q_major @ R_major
                W = Q_major @ R_major
                A = torch.split(Q_minor, config.rank, dim=1)
                B = torch.split(R_minor, config.rank, dim=0)

                new_layer.downscale_weights.from_weights(A)
                new_layer.upscale_weights.from_weights(B)

                new_layer.acc_downweight = nn.Parameter(W.to(config.device).contiguous(), requires_grad=False)

                for up_weight in new_layer.upscale_weights:
                    up_weight.require_grad = True
                for down_weight in new_layer.downscale_weights:
                    down_weight.require_grad = True
            elif config.decompose == 'keep':
                try:
                    new_layer.acc_downweight = nn.Parameter(module.weight.data.T.to(config.device).contiguous(), requires_grad=False)
                except:
                    new_layer.acc_downweight = nn.Parameter(torch.empty(module.weight.data.T.shape), requires_grad=False)
                    print(f"Found issue with copying module `{name}`, setting empty param...")

            if module.bias is not None:
                new_layer.bias = module.bias

            if convertion:
                new_layer.acc_downweight.to(weight_device).type(weight_type)
                new_layer.downscale_weights.to(weight_device).type(weight_type)
                new_layer.upscale_weights.to(weight_device).type(weight_type)
            
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, child_name, new_layer)
            else:
                setattr(model, name, new_layer)
            
            try:
                module.to("cpu")
            except:
                pass
            del module
            torch.cuda.empty_cache()
            
            pbar.update(1)

    return model

class SoWModel(PeftModel):
    def __init__(self, model, config: SoWConfig):
        # super().__init__(model, config)
        self.config = config
        self.model = prepare_sow(model, config)


def load_sow(model, checkpoint_path):
    """Load a model with SoW layers from a safetensor checkpoint file."""
    loaded_tensors = load_file(checkpoint_path)

    def get_nested_attr(model, name):
        parts = name.split(".")
        attr = model
        for part in parts:
            attr = getattr(attr, part)
        return attr
    
    for name, param in loaded_tensors.items():
        if name in model.state_dict():
            try:
                model_param = get_nested_attr(model, name)
            except AttributeError:
                print(f"Attribute '{name}' not found in model.")

            if model_param.numel() == 0:
                cloned_param = nn.Parameter(param.clone(), requires_grad=False)
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, child_name, cloned_param)
                else:
                    setattr(model, name, cloned_param)
            else:
                model_param.data.copy_(param.data)



def accumulate(model):
    for _, module in model.named_modules():
        if isinstance(module, SoWLinear):
            module.accumulate()

def export_alignment(module, export_name):
    if not isinstance(module, SoWLinear):
        raise TypeError("Not a SoW layer")
    
    accumalation = torch.sum(torch.stack([
        a.detach() @ b.detach() 
        for a, b in zip(module.downscale_weights, module.upscale_weights)
    ]), dim=0).detach()

    if module.acc_upweight.numel() != 0:
        weight = module.acc_downweight @ module.acc_upweight
    else:
        weight = module.acc_downweight

    U_acc, S_acc, V_acc = svd_weight(accumalation, module.rank)
    U_w, S_w, V_w = svd_weight(weight)

    alignment_grid = torch.abs(U_w.T @ U_acc)
    alignment_percentage = (alignment_grid / alignment_grid.sum(axis=0)) * 100

    alignment_percentage = alignment_percentage.detach().cpu().numpy()
    np.save(os.path.join('/home/antoine/code/tn_gradient/scripts/data/align', export_name+'.npy'), alignment_percentage)
