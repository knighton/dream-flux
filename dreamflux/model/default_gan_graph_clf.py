from torch import nn
from torch.optim import Adam

from ..layer import InstanceNorm, ShardedLinear
from .gan_graph_clf import GANGraphClassifier
from .util import int_interpolate


class GANBlock(nn.Sequential):
    def __init__(self, shard_dim, shard_in_dim, shard_out_dim):
        super().__init__(
            ShardedLinear(shard_dim, shard_in_dim, shard_out_dim),
            InstanceNorm(),
            nn.ReLU(),
        )


class ClfBlock(nn.Sequential):
    def __init__(self, in_dim, out_dim):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            InstanceNorm(),
            nn.ReLU(),
            nn.Dropout(),
        )


class DefaultGANGraphClassifier(GANGraphClassifier):
    def __init__(self, in_height, in_width, out_classes, embed_dim,
                 inputs_per_neuron, outputs_per_neuron, num_neurons, latent_dim,
                 ticks_per_sample):
        graph_optimizer = Adam

        a, b, c, d = int_interpolate(latent_dim, inputs_per_neuron, 4)
        g_model = nn.Sequential(
            GANBlock(num_neurons, a, b),
            GANBlock(num_neurons, b, c),
            ShardedLinear(num_neurons, c, d),
            nn.Tanh(),
        )
        g_optimizer = Adam

        a, b, c = int_interpolate(inputs_per_neuron, 1, 3)
        d_model = nn.Sequential(
            GANBlock(num_neurons, a, b),
            ShardedLinear(num_neurons, b, c),
            nn.Sigmoid(),
        )
        d_optimizer = Adam

        a, b, c, d = int_interpolate(num_neurons, out_classes, 4)
        clf = nn.Sequential(
            ClfBlock(a, b),
            ClfBlock(b, c),
            nn.Linear(c, d),
        )
        clf_optimizer = Adam

        super().__init__(in_height, in_width, out_classes, embed_dim,
                         inputs_per_neuron, outputs_per_neuron, num_neurons,
                         latent_dim, ticks_per_sample, graph_optimizer, g_model,
                         g_optimizer, d_model, d_optimizer, clf, clf_optimizer)
