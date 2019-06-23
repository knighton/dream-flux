import torch
from torch import nn
from torch.nn import Parameter as P


def dot_products_to_fractions(x, soft_clip, sharpness):
    # Normalize to a bell curve around zero of sources per destination.
    x = x - x.mean(1, keepdim=True)
    x = x / (x.std(1, keepdim=True) + 1e-3)

    # Apply a smooth cutoff to the scores.
    x = (x / soft_clip).tanh() * soft_clip

    # Translate those scores to weights.
    x = sharpness ** x

    # Translate the weights to fractions of sources per destination.
    return x / (x.sum(1, keepdim=True) + 1e-3)


class Core(nn.Module):
    """
    Core tick implementation (excludes I/O).
    """

    def __init__(self, meme_dim, num_nodes, vision_num_memes, inputs_per_node,
                 vocab_size, soft_clip, sharpness):
        super().__init__()

        self.meme_dim = meme_dim
        self.num_nodes = num_nodes
        self.vision_num_memes = vision_num_memes
        self.inputs_per_node = inputs_per_node
        self.vocab_size = vocab_size

        x = torch.Tensor([soft_clip])
        x = x.log()
        self.raw_soft_clip = P(x)

        x = torch.Tensor([sharpness])
        x = x.log()
        self.raw_sharpness = P(x)

        x = torch.randn(num_nodes, meme_dim)
        self.register_buffer('memes_from_last_tick', x)

        x = torch.randint(num_nodes + vision_num_memes, (num_nodes, inputs_per_node))
        self.register_buffer('input_indices', x)

        x = torch.randn(num_nodes, meme_dim, vocab_size)
        self.vocab = P(x)

    def spread(self, x):
        x = x[self.input_indices]
        x = x.mean(1)
        x = x - x.mean()
        return x / x.std()

    def remix(self, x):
        dot_products = torch.einsum('ne,nev->nv', [x, self.vocab])
        soft_clip = self.raw_soft_clip.exp()
        sharpness = self.raw_sharpness.exp()
        fractions = dot_products_to_fractions(dot_products, soft_clip, sharpness)
        return torch.einsum('nev,nv->ne', [self.vocab, fractions])

    def forward(self, x):
        memes_from_input = x
        x = torch.cat([memes_from_input, self.memes_from_last_tick], 0)
        x = self.spread(x)
        x = self.remix(x)
        self.memes_from_last_tick = x.detach()
        return x
