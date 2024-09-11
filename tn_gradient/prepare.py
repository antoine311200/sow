import torch
import torch.nn as nn

from tn_gradient.layer.sow import SoWLinear, SoWArgs


def prepare_sow(
    model, target_modules, decompose: bool = True, args: SoWArgs = SoWArgs()
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
        model (_type_): _description_
        target_modules (_type_): _description_
        decompose (bool, optional): _description_. Defaults to True.
        args (SoWArgs, optional): _description_. Defaults to SoWArgs().

    Returns:
        _type_: _description_
    """
    layers_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            layers_to_replace[name] = module

    for name, module in layers_to_replace.items():
        convertion = False
        if module.weight.data.dtype != torch.float:
            convertion = True

            weight_type = module.weight.data.dtype
            weight_device = module.weight.data.device
            module.weight = module.weight.to(torch.float)

        new_layer = SoWLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=args.rank,
            n_iter=args.n_iter,
            bias=module.bias is not None,
            dtype=module.weight.data.dtype,
            device=args.device,
        )

        if decompose:
            keep_rank = args.rank * args.n_iter
            Q, R = torch.linalg.qr(module.weight.data.T)
            Q_major, Q_minor = (
                Q[:, :-keep_rank],
                Q[:, -keep_rank:],
            )
            R_major, R_minor = (
                R[:-keep_rank, :],
                R[-keep_rank:, :],
            )

            W = Q_major @ R_major
            A = torch.split(Q_minor, args.rank, dim=1)
            B = torch.split(R_minor, args.rank, dim=0)

            new_layer.downscale_weights.from_weights(A)
            new_layer.upscale_weights.from_weights(B)
            new_layer.accumulated_weight = W.to(args.device)

        if module.bias is not None:
            new_layer.bias = module.bias

        if convertion:
            new_layer.accumulated_weight.to(weight_device).type(weight_type)
            new_layer.downscale_weights.to(weight_device).type(weight_type)
            new_layer.upscale_weights.to(weight_device).type(weight_type)

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)
        else:
            setattr(model, name, new_linear)

    return model
