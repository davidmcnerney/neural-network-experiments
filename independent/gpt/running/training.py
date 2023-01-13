import statistics
from typing import List, Set, Tuple

import torch
from torch import nn
import torch.nn.utils
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
    return torch.optim.AdamW(groups, lr=model.config.learning_rate)


def train(
    model: GPT,
    dataset: torch.utils.data.Dataset,
) -> None:
    # TODO: set num_workers for multiprocessing in data loader
    # TODO: make sure we are on the right devices everywhere
    #    The code for ^ can be found in minGPT and nanoGPT

    optimizer = get_optimizer(model)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=model.config.batch_size,
    )

    for epoch_num in range(model.config.count_epochs):
        epoch_losses: List[float] = []
        for batch in iter(loader):
            x, y = batch
            logits = model(x)
            loss = model.calculate_loss(logits, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad(model.parameters(), model.config.grad_norm_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch_num} loss {statistics.mean(epoch_losses)}")

    print("Training complete")


def _parameters_by_weight_decay_requirement(model: GPT) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Bias parameters, layer normalization weights, and embedding weights are not to get weight decay.
    """

    names_requiring_weight_decay = set()
    names_not_requiring_weight_decay = set()
    for module_name, module in model.named_modules():
        for parameter_name, _ in module.named_parameters():
            full_parameter_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if parameter_name.endswith("bias"):
                names_not_requiring_weight_decay.add(full_parameter_name)
            elif parameter_name.endswith("weight"):
                if isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    names_not_requiring_weight_decay.add(full_parameter_name)
                elif isinstance(module, nn.Linear):
                    names_requiring_weight_decay.add(full_parameter_name)
                else:
                    # We end up here when we encounter parameters in our container modules as we recursively descend,
                    # because .named_parameters() is recursive. We'll eventually encounter each parameter in the context
                    # of a LayerNorm, Embedding, or Linear Pytorch module, and categorize it then.
                    pass
            else:
                raise Exception(f"Parameter {parameter_name} does not end with expected `weight` or `bias`")

    parameters_by_name = {name: parameter for name, parameter in model.named_parameters()}
    # _summarize_set("names_requiring_weight_decay", names_requiring_weight_decay)
    # _summarize_set("names_not_requiring_weight_decay", names_not_requiring_weight_decay)
    # _summarize_set("parameters_by_name", set(parameters_by_name.keys()))
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


def _summarize_set(description: str, input_set: Set):
    as_list = sorted(input_set)
    print(f"\n{description} ({len(input_set)}):")
    for item in as_list:
        print(f"   {item}")
    print("")
