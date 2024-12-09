import torch

from tn_gradient.layer.sow import SoWLinear

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
    memory_usage = 0#sum(p.numel() for p in model.parameters())
    buffer_size = 0
    for param in model.parameters():
        # if isinstance(param, torch.Tensor):
        memory_usage += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    for _, module in model.named_modules():
        if isinstance(module, SoWLinear):
            for weight in module.upscale_weights:
                memory_usage_sow += weight.nelement() * weight.element_size()
            for weight in module.downscale_weights:
                memory_usage_sow += weight.nelement() * weight.element_size()
            
            memory_usage_accum += module.in_features * module.out_features
    
    return memory_usage, memory_usage_sow, memory_usage_accum, buffer_size
