from typing import List, Tuple

import torch
from torch import nn
import torch.optim
import torch.utils.data

from independent.gpt.model.gpt import GPT


def get_optimizer(model: GPT) -> torch.optim.Optimizer:
    """
    We configure an AdamW optimizer, with no weight decay for biases, layer normalization weights,
    and embedding weights.
    """
    parameters_requiring_weight_decay, parameters_not_requiring_weight_decay = _parameters_by_weight_decay_requirement(model)
    groups = [
        {"params": parameters_requiring_weight_decay, "weight_decay": model.config.weight_decay},
        {"params": parameters_not_requiring_weight_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, learning_rate=model.config.learning_rate)


def train(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
) -> None:
    pass


def _parameters_by_weight_decay_requirement(model: GPT) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    names_requiring_weight_decay = set()
    names_not_requiring_weight_decay = set()
    for module_name, module in model.named_modules():
        for parameter_name, _ in module.named_parameters():
            full_parameter_name = f"{module_name}.{parameter_name}" if module_name is not None else parameter_name
            if parameter_name.endswith("bias"):
                names_not_requiring_weight_decay.add(full_parameter_name)
            elif parameter_name.endswith("weight"):
                if isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    names_not_requiring_weight_decay.add(full_parameter_name)
                elif isinstance(module, nn.Linear):
                    names_requiring_weight_decay.add(full_parameter_name)
                else:
                    # We end up here when we encounter parameters in our container modules as we recursively descend,
                    # because .named_parameters() is recursive.
                    pass
            else:
                raise Exception(f"Parameter {parameter_name} does not end with expected `weight` or `bias`")

    parameters_by_name = {name: parameter for name, parameter in module.named_parameters()}
    if len(parameters_by_name) != len(names_requiring_weight_decay) + len(names_not_requiring_weight_decay):
        raise Exception
    requiring_weight_decay = [
        parameters_by_name[parameter_name]
        for parameter_name in sorted(names_requiring_weight_decay)
    ]
    not_requiring_weight_decay = [
        parameters_by_name[parameter_name]
        for parameter_name in sorted(names_not_requiring_weight_decay)
    ]

    return requiring_weight_decay, not_requiring_weight_decay
