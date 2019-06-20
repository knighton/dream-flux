from time import time
import torch
from torch.nn import functional as F
from torch.nn import Parameter as P

from .model import Model
from .util import sharded_binary_cross_entropy


class Info(object):
    fields = 'graph_time', 'graph_std_loss', 'graph_clf_loss', 'graph_clf_acc', \
        'd_time', 'd_real_loss', 'd_real_isreal', 'd_fake_loss', 'd_fake_isreal', \
        'g_time', 'g_loss'

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
        self.graph_time = None
        self.graph_std_loss = None
        self.graph_clf_loss = None
        self.graph_clf_acc = None

        self.d_time = None
        self.d_real_loss = None
        self.d_real_isreal = None
        self.d_fake_loss = None
        self.d_fake_isreal = None

        self.g_time = None
        self.g_loss = None

    def dump(self):
        return self.__dict__

    def to_text(self):
        lines = [
            '* Time',
            '  * Graph                %5.1fms' % (1000 * self.graph_time,),
            '  * Discriminator        %5.1fms' % (1000 * self.d_time,),
            '  * Generator            %5.1fms' % (1000 * self.g_time,),
            '* Loss',
            '  * Graph (std)          %7.3f' % self.graph_std_loss,
            '  * Graph (clf)          %7.3f' % self.graph_clf_loss,
            '  * Discriminator (real) %7.3f' % self.d_real_loss,
            '  * Discriminator (fake) %7.3f' % self.d_fake_loss,
            '  * Generator            %7.3f' % self.g_loss,
            '* Is real',
            '  * Discriminator (real) %7.3f' % self.d_real_isreal,
            '  * Discriminator (fake) %7.3f' % self.d_fake_isreal,
            '* Accuracy',
            '  * Graph                %6.2f%%' % (100 * self.graph_clf_acc,),
        ]
        return ''.join(map(lambda line: line + '\n', lines))


class GANGraphProber(Model):
    def __init__(self, in_height, in_width, out_classes, embed_dim,
                 inputs_per_neuron, outputs_per_neuron, num_neurons, latent_dim,
                 ticks_per_sample, graph_optimizer, g, g_optimizer, d, d_optimizer):
        super().__init__()

        # Configuration.
        in_dim = in_height * in_width
        self.in_height = in_height
        self.in_width = in_width
        self.in_dim = in_dim
        self.out_classes = out_classes
        self.embed_dim = embed_dim
        self.inputs_per_neuron = inputs_per_neuron
        self.outputs_per_neuron = outputs_per_neuron
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim
        self.ticks_per_sample = ticks_per_sample

        # State (sent to next tick).
        #
        # - Activations (num neurons).
        self.register_buffer('activations', torch.rand(num_neurons))

        # Graph (ie, embeddings).
        #
        # - Input sinks (embed dim, outputs per neuron, in dim).
        # - Neuron sinks (embed dim, outputs per neuron, num neurons).
        # - Neuron sources (embed dim, inputs per neuron, num neurons).
        # - Output sources (embed dim, inputs per neuron, out classes).
        x = self.make_input_sinks(in_height, in_width, embed_dim)
        x = x.unsqueeze(1)
        x = x.repeat(1, outputs_per_neuron, 1)
        self.input_sinks = P(x)
        self.neuron_sinks = P(torch.randn(
            embed_dim, outputs_per_neuron, num_neurons))
        self.neuron_sources = P(torch.randn(
            embed_dim, inputs_per_neuron, num_neurons))
        parameters = self.input_sinks, self.neuron_sinks, self.neuron_sources
        self.graph_optimizer = graph_optimizer(parameters)

        self.output_sources = P(torch.randn(
            embed_dim, inputs_per_neuron, out_classes))
        self.clf_optimizer = graph_optimizer([self.output_sources])

        # Neurons (ie, a sharded GAN).
        #
        # - Generator (latent dim, num neurons) -> (inputs per neuron, num neurons).
        # - Discriminator (inputs per neuron, num neurons) -> (1, num neurons).
        self.g = g
        self.g_optimizer = g_optimizer(g.parameters())
        self.d = d
        self.d_optimizer = d_optimizer(d.parameters())

    @classmethod
    def make_input_sinks(cls, in_height, in_width, embed_dim):
        return torch.randn(embed_dim, in_height * in_width)

        assert 2 <= embed_dim

        ones = torch.randn(embed_dim - 2, in_height * in_width) * 0.05 + 1

        y = torch.arange(in_height).type(torch.float) / (in_height - 1)
        y = y * 2 - 1
        y = y.unsqueeze(1)
        y = y.repeat(1, in_width)
        y = y.view(1, -1)

        x = torch.arange(in_width).type(torch.float) / (in_width - 1)
        x = x * 2 - 1
        x = x.unsqueeze(0)
        x = x.repeat(in_height, 1)
        x = x.view(1, -1)

        return torch.cat([ones, y, x], 0)

    def project(self, values, from_locations, to_locations):
        # Relate the source and destination vectors.
        from_ = from_locations.view(self.embed_dim, -1)
        to = to_locations.view(self.embed_dim, -1)
        io = torch.einsum('ei,eo->io', [from_, to])

        # Normalize scores to a bell curve around zero of sources per destination.
        io = io - io.mean(1, keepdim=True)
        io = io / (io.std(1, keepdim=True) + 1e-3)

        # Apply a smooth cutoff to the scores (-3, 3).
        io = (io / 3).tanh() * 3

        # Translate that to weights (0.05, 20).
        io = io.exp()

        # Make those weights fractions (of sources per destination).
        io = io / (io.sum(1, keepdim=True) + 1e-3)

        # Unpack shapes.
        io = io.view(self.outputs_per_neuron, self.in_dim + self.num_neurons,
                     self.inputs_per_neuron, -1)#self.num_neurons + self.out_classes)

        # Do the projection.
        x = torch.einsum('a,oaib->ib', [values, io])
        return x.sigmoid()

    def train_graph(self, x, y, info):
        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        self.graph_optimizer.zero_grad()

        inputs = F.dropout(x.view(-1), 0.8)
        inputs = inputs / (inputs.max() + 1e-3)
        activ_at_sinks = torch.cat([inputs, self.activations], 0)
        sinks = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        activ_at_sources = self.project(activ_at_sinks, sinks, self.neuron_sources)

        loss = activ_at_sources.std(0).mean()
        loss.backward()
        self.graph_optimizer.step()
        info.graph_std_loss = loss.item()

        self.clf_optimizer.zero_grad()
        y_pred = self.project(activ_at_sinks.detach(), sinks.detach(),
                              self.output_sources)
        y_pred = y_pred.mean(0, keepdim=True)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        self.clf_optimizer.step()
        info.graph_clf_loss = loss.item()

        info.graph_clf_acc = (y_pred[0].argmax() == y).type(torch.float32).item()

        info.graph_time = time() - t0

        return activ_at_sources.detach(), y_pred.detach()

    def validate_graph(self, x, y, info):
        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        inputs = F.dropout(x.view(-1), 0.8)
        inputs = inputs / (inputs.max() + 1e-3)
        x = torch.cat([inputs, self.activations], 0)
        sinks = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        sources = torch.cat([self.neuron_sources, self.output_sources], 2)
        x = self.project(x, sinks, sources)

        loss = x.std(0).mean()
        info.graph_std_loss = loss.item()

        y_pred = x[:, :self.out_classes]
        y_pred = y_pred.mean(0, keepdim=True)
        loss = F.cross_entropy(y_pred, y)
        info.graph_clf_loss = loss.item()

        info.graph_clf_acc = (y_pred[0].argmax() == y).type(torch.float32).item()

        info.graph_time = time() - t0

        return x[:, :-self.out_classes].detach(), y_pred.detach()

    def forward_graph(self, x):
        assert x.shape == (1, self.in_height, self.in_width)

        inputs = F.dropout(x.view(-1), 0.8)
        inputs = inputs / (inputs.max() + 1e-3)
        x = torch.cat([inputs, self.activations], 0)
        sinks = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        sources = torch.cat([self.neuron_sources, self.output_sources], 2)
        x = self.project(x, sinks, sources)

        y_pred = x[:, :self.out_classes]
        y_pred = y_pred.mean(0, keepdim=True)

        return x[:, :-self.out_classes].detach(), y_pred.detach()

    def train_neurons(self, x, info):
        t0 = time()

        self.d_optimizer.zero_grad()

        real = x.unsqueeze(0)
        target = torch.full((1, 1, self.num_neurons), 1, device=x.device)
        is_real = self.d(real)
        loss = sharded_binary_cross_entropy(is_real, target).mean()
        loss.backward()
        ret = is_real.detach()
        info.d_real_loss = loss.item()
        info.d_real_isreal = is_real.detach().mean().item()

        latent = torch.randn(1, self.latent_dim, self.num_neurons, device=x.device)
        fake = self.g(latent)
        target.fill_(0)
        is_real = self.d(fake.detach())
        loss = sharded_binary_cross_entropy(is_real, target).mean()
        loss.backward()
        info.d_fake_loss = loss.item()
        info.d_fake_isreal = is_real.detach().mean().item()

        self.d_optimizer.step()

        t1 = time()

        self.g_optimizer.zero_grad()
        target.fill_(1)
        is_real = self.d(fake)
        loss = sharded_binary_cross_entropy(is_real, target).mean()
        loss.backward()
        self.g_optimizer.step()
        info.g_loss = loss.item()

        t2 = time()
        info.d_time = t1 - t0
        info.g_time = t2 - t1

        return ret.squeeze()

    def validate_neurons(self, x, info):
        t0 = time()

        real = x.unsqueeze(0)
        target = torch.full((1, 1, self.num_neurons), 1, device=x.device)
        is_real = self.d(real)
        loss = sharded_binary_cross_entropy(is_real, target).mean()
        ret = is_real.detach()
        info.d_real_loss = loss.item()
        info.d_real_isreal = is_real.detach().mean().item()

        latent = torch.randn(1, self.latent_dim, self.num_neurons, device=x.device)
        fake = self.g(latent)
        target.fill_(0)
        is_real = self.d(fake.detach())
        loss = sharded_binary_cross_entropy(is_real, target).mean()
        info.d_fake_loss = loss.item()
        info.d_fake_isreal = is_real.detach().mean().item()

        self.d_optimizer.step()

        t1 = time()

        target.fill_(1)
        is_real = self.d(fake)
        loss = sharded_binary_cross_entropy(is_real, target).mean()
        self.g_optimizer.step()
        info.g_loss = loss.item()

        t2 = time()
        info.d_time = t1 - t0
        info.g_time = t2 - t1

        return ret.squeeze()

    def forward_neurons(self, x):
        return self.d(x).detach().squeeze(0)

    def train_on_tick(self, x, y):
        info = Info()
        a, y_pred = self.train_graph(x, y, info)
        self.activations = self.train_neurons(a, info)
        return y_pred, info

    def validate_on_tick(self, x, y):
        info = Info()
        a, y_pred = self.validate_graph(x, y, info)
        self.activations = self.train_neurons(a, info)
        return y_pred, info

    def forward_on_tick(self, x):
        a, y_pred = self.forward_graph(x)
        self.activations = self.forward_neurons(a)
        return y_pred
