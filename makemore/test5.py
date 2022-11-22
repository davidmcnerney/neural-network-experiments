from typing import List, Optional

import torch


generator = torch.Generator().manual_seed(2147483647)


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = torch.randn((in_features, out_features), generator=generator)
        if bias:
            self.bias = torch.zeros(out_features)
        self.out: Optional[torch.tensor] = None

    def __call__(self, x) -> torch.Tensor:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        return_value = [self.weight]
        if self.bias is not None:
            return_value.append(self.bias)
        return return_value


class BatchNorm1d:
    def __init__(self, dim, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps            # small constant to prevent divide by zero errors
        self.momentum = momentum  # how quickly running_mean and running_var update

        self.training = True

        # Trainable parameters that tune the normalization
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # We capture average mean and variance while training, for use in inference later
        self.running_mean = torch.zeros(dim)
        self.running_variance = torch.ones(dim)

    def __call__(self, x) -> torch.Tensor:
        if self.training:
            # Compute batch mean and variance during training
            mean = x.mean(dim=0, keepdim=True)
            variance = x.var(dim=0, keepdim=True)
        else:
            # Use previously captured average mean and variance during inference
            mean = self.running_mean
            variance = self.running_variance

        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        self.out = self.gamma * x_normalized + self.beta

        # Update our running mean and variance
        if self.training:
            with torch.no_grad():
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_variance = (1.0 - self.momentum) * self.running_variance + self.momentum * variance

        return self.out

    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        return []
