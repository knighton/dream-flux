from torch import nn


def instance_norm(x, eps=1e-6):
    x = x - x.mean()
    return x / (x.std() + eps)


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return instance_norm(x, self.eps)
