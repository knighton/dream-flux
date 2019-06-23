from time import time
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.optim import Adam

from ..layer import *
from .tick_model import TickModel


class Vision(nn.Module):
    """
    Processes images, converting them into sets of memes.
    """

    def __init__(self, in_shape, body_channels, out_memes, meme_dim):
        super().__init__()

        in_channels, in_height, in_width = in_shape

        assert in_channels == 1
        assert in_height == 28
        assert in_width == 28

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.body_channels = body_channels
        self.out_memes = out_memes
        self.meme_dim = meme_dim

        c = body_channels
        self.seq = nn.Sequential(
            # 28x28.
            nn.Conv2d(1, c, 3, 1, 0),
            InstanceNorm(),

            # 26x26.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            InstanceNorm(),

            # 24x24.
            nn.MaxPool2d(2),

            # 12x12.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            InstanceNorm(),

            # 10x10.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            InstanceNorm(),

            # 8x8.
            nn.MaxPool2d(2),

            # 4x4.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            InstanceNorm(),

            # 4x4.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            InstanceNorm(),

            Flatten(),

            # 16.
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16 * c, out_memes * meme_dim),
            InstanceNorm(),
            Reshape(out_memes, meme_dim),
        )

    def forward(self, x):
        """
        Process an input image into meme vectors.

        Input
        - x (in channels, in height, in width)

        Output
        - memes (num input memes, meme dim)
        """
        assert x.shape == (self.in_channels, self.in_height, self.in_width)
        return self.seq(x.unsqueeze(0)).squeeze(0)


class DenseBlock(nn.Sequential):
    """
    Linear/Norm/ReLU/Dropout.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            InstanceNorm(),
            nn.ReLU(),
            nn.Dropout(),
        )


class Inception(nn.Sequential):
    """
    Generate a meme from a seed (like a GAN generator).
    """

    def __init__(self, seed_dim, meme_dim):
        a = seed_dim
        d = meme_dim
        b = (a * 2 + d * 1) // 3
        c = (a * 1 + d * 2) // 3
        super().__init__(
            DenseBlock(a, b),
            DenseBlock(b, c),
            nn.Linear(c, d),
            InstanceNorm(),
        )


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


class Graph(nn.Module):
    def __init__(self, num_nodes, receivers_per_node, receiver_seed_dim,
                 receiver_seed_noise, meme_dim, soft_clip, sharpness):
        super().__init__()

        self.num_nodes = num_nodes
        self.receivers_per_node = receivers_per_node
        self.receiver_seed_dim = receiver_seed_dim
        self.receiver_seed_noise = receiver_seed_noise
        self.meme_dim = meme_dim

        x = torch.Tensor([soft_clip])
        x = x.log()
        self.raw_soft_clip = P(x)

        x = torch.Tensor([sharpness])
        x = x.log()
        self.raw_sharpness = P(x)

        self.inception = Inception(receiver_seed_dim, meme_dim)

    def make_receiver_seeds(self, device):
        """
        Create random seeds for generating groups of similar memes.

        Input
        - device

        Output
        - seeds (num nodes, receivers per node, meme dim)
        """
        shape = self.num_nodes, 1, self.receiver_seed_dim
        base = torch.randn(*shape, device=device)
        base = base.repeat(1, self.receivers_per_node, 1)
        shape = self.num_nodes, self.receivers_per_node, \
            self.receiver_seed_dim
        noise = torch.randn(*shape, device=device) * self.receiver_seed_noise
        return base + noise

    def generate_receivers(self, x):
        """
        Generate the receiver memes.

        Input
        - x (num input memes + num nodes, meme dim)

        Output
        - receivers (num nodes, receivers per node, meme dim)
        """
        seeds = self.make_receiver_seeds(x.device)
        seeds = seeds.view(-1, self.receiver_seed_dim)
        memes = self.inception(seeds)
        return memes.view(self.num_nodes, self.receivers_per_node,
                          self.meme_dim)

    def activate(self, senders, receivers):
        """
        Activate receiver memes, given sender memes.

        Input
        - senders (num input memes + num nodes, meme dim)
        - receivers (num nodes, receivers per node, meme dim)

        Output
        - fractions (num nodes, receivers per node)
        """
        # Relate the source and destination meme vectors (dot product).
        x = torch.einsum('me,nre->mnr', [senders, receivers])

        # Reshape to (in memes, out memes).
        x = x.view(-1, self.num_nodes * self.receivers_per_node)

        # Convert the dot products to fractional weights.
        soft_clip = self.raw_soft_clip.exp()
        sharpness = self.raw_sharpness.exp()
        x = dot_products_to_fractions(x, soft_clip, sharpness)

        # Reduce on the sources.
        x = x.sum(0)

        # Unpack output axes.
        return x.view(self.num_nodes, self.receivers_per_node)

    def forward(self, x):
        """
        Activate the receivers and combine according to their activations.

        Input
        - senders (num input memes + num nodes, meme dim)
        - receivers (num nodes, receivers per node, meme dim)

        Output
        - reco_inputs (num nodes, meme dim)
        """
        receivers = self.generate_receivers(x)

        # Get activations (as fractions of contribution).
        fractions = self.activate(x, receivers)

        # Reduce-sum on that dimension.
        return torch.einsum('nre,nr->ne', [receivers, fractions])


class Reconstructor(nn.Module):
    """
    Reconstruct the input memes using our own internal vocabulary of memes.
    """

    def __init__(self, num_nodes, meme_dim, vocab_size, soft_clip, sharpness):
        super().__init__()

        self.num_nodes = num_nodes
        self.meme_dim = meme_dim
        self.vocab_size = vocab_size

        x = torch.Tensor([soft_clip])
        x = x.log()
        self.raw_soft_clip = P(x)

        x = torch.Tensor([sharpness])
        x = x.log()
        self.raw_sharpness = P(x)

        x = torch.randn(num_nodes, meme_dim, vocab_size)
        self.vocab = P(x)

    def forward(self, x):
        weights = torch.einsum('ne,nev->nv', [x, self.vocab])
        soft_clip = self.raw_soft_clip.exp()
        sharpness = self.raw_sharpness.exp()
        fractions = dot_products_to_fractions(weights, soft_clip, sharpness)
        return torch.einsum('nev,nv->ne', [self.vocab, fractions])


class MemeWeighter(nn.Sequential):
    """
    Weight memes for combining into a summary.
    """

    def __init__(self, meme_dim):
        a = meme_dim
        b = meme_dim // 2
        c = meme_dim // 4
        d = 1
        super().__init__(
            DenseBlock(a, b),
            DenseBlock(b, c),
            nn.Linear(c, d),
            InstanceNorm(),
        )


class MemeClassifier(nn.Sequential):
    """
    Classify a single meme summarizing the population of them.
    """

    def __init__(self, meme_dim, num_classes):
        a = meme_dim
        b = meme_dim * 2
        c = meme_dim * 4
        d = num_classes
        super().__init__(
            DenseBlock(a, b),
            DenseBlock(b, c),
            nn.Linear(c, d),
        )


class Classifier(nn.Module):
    """
    Classify the sample given its resulting population of memes.
    """

    def __init__(self, meme_dim, num_classes):
        super().__init__()
        self.meme_dim = meme_dim
        self.num_classes = num_classes
        self.weight_memes = MemeWeighter(meme_dim)
        self.classify_meme = MemeClassifier(meme_dim, num_classes)

    def forward(self, x):
        """
        Given the reconstructed memes, classify the sample.

        Input
        - memes (num nodes, meme dim)

        Output
        - classes (num classes)
        """
        # Do some magic to assign weights to the meme vectors.
        weights = self.weight_memes(x).squeeze(1)

        # Combine the memes into a summary according to the weights.
        x = torch.einsum('ne,n->e', [x, weights])

        # Normalize.
        x = x - x.mean()
        x = x / (x.std() + 1e-3)

        # Classify the summary.
        return self.classify_meme(x)


class RecoGraphCore(nn.Module):
    """
    A classifier that forwards on single samples.
    """

    def __init__(self, in_shape, vision_body_channels, vision_out_memes,
                 meme_dim, num_nodes, receivers_per_node, receiver_seed_dim,
                 receiver_seed_noise, vocab_size, transmit_soft_clip,
                 transmit_sharpness, remix_soft_clip, remix_sharpness,
                 num_classes):
        super().__init__()

        in_channels, in_height, in_width = in_shape
        self.in_shape = in_shape
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.vision_body_channels = vision_body_channels
        self.vision_out_memes = vision_out_memes

        self.meme_dim = meme_dim
        self.num_nodes = num_nodes
        self.receivers_per_node = receivers_per_node
        self.receiver_seed_dim = receiver_seed_dim
        self.vocab_size = vocab_size

        self.transmit_soft_clip = transmit_soft_clip
        self.transmit_sharpness = transmit_sharpness
        self.remix_soft_clip = remix_soft_clip
        self.remix_sharpness = remix_sharpness

        x = torch.randn(num_nodes, meme_dim)
        self.register_buffer('memes_from_last_tick', x)

        self.observe = Vision(
            in_shape, vision_body_channels, vision_out_memes, meme_dim)

        self.transmit = Graph(
            num_nodes, receivers_per_node, receiver_seed_dim,
            receiver_seed_noise, meme_dim, transmit_soft_clip,
            transmit_sharpness)

        self.remix = Reconstructor(
            num_nodes, meme_dim, vocab_size, remix_soft_clip, remix_sharpness)

        self.classify = Classifier(meme_dim, num_classes)

    def forward(self, x):
        """
        Perform one timestep of work.  Learns on the basis of ticks.

        Input
        - x (in channels, in height, in width)

        Output
        - y (out classes)
        """
        memes_from_input = self.observe(x)
        memes = torch.cat([memes_from_input, self.memes_from_last_tick], 0)
        memes = self.transmit(memes)
        memes = self.remix(memes)
        self.memes_from_last_tick = memes.detach()
        return self.classify(memes)


class Info(object):
    fields = 'time', 'loss', 'acc'

    @classmethod
    def mean(cls, infos):
        ret = Info()

        for field in cls.fields:
            setattr(ret, field, [])

        for info in infos:
            for field in cls.fields:
                value = getattr(info, field)
                getattr(ret, field).append(value)

        for field in cls.fields:
            values = getattr(ret, field)
            setattr(ret, field, sum(values) / len(values))

        return ret

    def __init__(self):
        self.time = None
        self.loss = None
        self.acc = None

    def dump(self):
        return self.__dict__

    def to_text(self):
        lines = [
            '* Time     %5.1fms' % (1000 * self.time,),
            '* Loss     %7.3f' % self.loss,
            '* Accuracy %6.2f%%' % (100 * self.acc,),
        ]
        return ''.join(map(lambda line: line + '\n', lines))


class RecoGraph(TickModel):
    """
    Wrapper to deal with the weird API (due to needs of other models).
    """

    def __init__(self, ticks_per_sample, in_shape, vision_body_channels,
                 vision_out_memes, meme_dim, num_nodes, receivers_per_node,
                 receiver_seed_dim, receiver_seed_noise, vocab_size,
                 transmit_soft_clip, transmit_sharpness, remix_soft_clip,
                 remix_sharpness, num_classes):
        super().__init__(ticks_per_sample)

        self.core = RecoGraphCore(
            in_shape, vision_body_channels, vision_out_memes, meme_dim,
            num_nodes, receivers_per_node, receiver_seed_dim,
            receiver_seed_noise, vocab_size, transmit_soft_clip,
            transmit_sharpness, remix_soft_clip, remix_sharpness, num_classes)

        self.optimizer = Adam(self.core.parameters())

    def forward_on_tick(self, x):
        return self.core(x)

    def train_on_tick(self, x, y_true):
        info = Info()
        t0 = time()
        self.optimizer.zero_grad()
        y_pred = self.core(x)
        y_pred = y_pred.unsqueeze(0)
        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred, info

    def validate_on_tick(self, x, y_true):
        info = Info()
        t0 = time()
        y_pred = self.core(x)
        y_pred = y_pred.unsqueeze(0)
        loss = F.cross_entropy(y_pred, y_true)
        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred.detach(), info
