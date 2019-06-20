import torch
from torch import nn
from torch.nn import Parameter as P


class ShardedLinear(nn.Module):
    def __init__(self, shard_dim, shard_in_dim, shard_out_dim):
        super().__init__()

        self.shard_dim = shard_dim
        self.shard_in_dim = shard_in_dim
        self.shard_out_dim = shard_out_dim

        self.pre_bias = P(torch.randn(shard_in_dim, shard_dim) * 0.05)
        self.kernel = P(torch.randn(shard_in_dim, shard_out_dim, shard_dim) * 0.05)
        self.post_bias = P(torch.randn(shard_out_dim, shard_dim) * 0.05)

    def forward(self, x):
        x = x + self.pre_bias
        x = torch.einsum('nis,ios->nos', [x, self.kernel])
        return x + self.post_bias
