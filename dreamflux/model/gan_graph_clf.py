from time import time
import torch
from torch.nn import functional as F
from torch.nn import Parameter as P

from .tick_model import TickModel
from .util import sharded_binary_cross_entropy


class Info(object):
    fields = 'graph_time', 'graph_loss', 'd_time', 'd_real_loss', 'd_real_isreal', \
        'd_fake_loss', 'd_fake_isreal', 'g_time', 'g_loss', 'clf_time', 'clf_loss', \
        'clf_acc'

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
        self.graph_loss = None

        self.d_time = None
        self.d_real_loss = None
        self.d_real_isreal = None
        self.d_fake_loss = None
        self.d_fake_isreal = None

        self.g_time = None
        self.g_loss = None

        self.clf_time = None
        self.clf_loss = None
        self.clf_acc = None

    def dump(self):
        return self.__dict__

    def to_text(self):
        lines = [
            '* Time',
            '  * Graph                %5.1fms' % (1000 * self.graph_time,),
            '  * Discriminator        %5.1fms' % (1000 * self.d_time,),
            '  * Generator            %5.1fms' % (1000 * self.g_time,),
            '  * Classifier           %5.1fms' % (1000 * self.clf_time,),
            '* Loss',
            '  * Graph                %7.3f' % self.graph_loss,
            '  * Discriminator (real) %7.3f' % self.d_real_loss,
            '  * Discriminator (fake) %7.3f' % self.d_fake_loss,
            '  * Generator            %7.3f' % self.g_loss,
            '  * Classifier           %7.3f' % self.clf_loss,
            '* Is real',
            '  * Discriminator (real) %7.3f' % self.d_real_isreal,
            '  * Discriminator (fake) %7.3f' % self.d_fake_isreal,
            '* Accuracy',
            '  * Classifier           %6.2f%%' % (100 * self.clf_acc,),
        ]
        return ''.join(map(lambda line: line + '\n', lines))


class GANGraphClassifier(TickModel):
    def __init__(self, in_height, in_width, out_classes, embed_dim,
                 inputs_per_neuron, outputs_per_neuron, num_neurons, latent_dim,
                 ticks_per_sample, graph_optimizer, g, g_optimizer, d, d_optimizer,
                 clf, clf_optimizer):
        super().__init__(ticks_per_sample)

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
        self.register_buffer('activations', torch.randn(num_neurons))

        # Graph (ie, embeddings).
        #
        # - Input sinks (embed dim, outputs per neuron, in dim).
        # - Neuron sinks (embed dim, outputs per neuron, num neurons).
        # - Neuron sources (embed dim, inputs per neuron, num neurons).
        x = self.make_input_sinks(in_height, in_width, embed_dim)
        x = x.unsqueeze(1)
        x = x.repeat(1, outputs_per_neuron, 1)
        self.input_sinks = P(x)
        self.neuron_sinks = P(torch.randn(embed_dim, outputs_per_neuron,
                                          num_neurons))
        self.neuron_sources = P(torch.randn(embed_dim, inputs_per_neuron,
                                            num_neurons))
        parameters = self.input_sinks, self.neuron_sinks, self.neuron_sources
        self.graph_optimizer = graph_optimizer(parameters)

        # Neurons (ie, a sharded GAN).
        #
        # - Generator (latent dim, num neurons) -> (inputs per neuron, num neurons).
        # - Discriminator (inputs per neuron, num neurons) -> (1, num neurons).
        self.g = g
        self.g_optimizer = g_optimizer(g.parameters())
        self.d = d
        self.d_optimizer = d_optimizer(d.parameters())

        # Output (ie, a model).
        #
        # - Model: (num neurons) -> (clf dim).
        self.clf = clf
        self.clf_optimizer = clf_optimizer(clf.parameters())

    @classmethod
    def make_input_sinks(cls, in_height, in_width, embed_dim):
        assert 2 <= embed_dim

        ones = torch.ones(embed_dim - 2, in_height * in_width)

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
        from_ = from_locations.view(self.embed_dim, -1)
        to = to_locations.view(self.embed_dim, -1)
        io = torch.einsum('ei,eo->io', [from_, to])
        io = io - io.mean(1, keepdim=True)
        io = io / io.std(1, keepdim=True)
        io = io.exp()
        io = io / io.sum(1, keepdim=True)

        io = io.view(self.outputs_per_neuron, self.in_dim + self.num_neurons,
                     self.inputs_per_neuron, self.num_neurons)

        return torch.einsum('a,oaib->ib', [values, io])

    def train_graph(self, x, info):
        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        self.graph_optimizer.zero_grad()

        x = torch.cat([x.view(-1), self.activations], 0)
        locations = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        x = self.project(x, locations, self.neuron_sources)

        loss = x.std(0).mean()
        loss.backward()
        self.graph_optimizer.step()

        info.graph_time = time() - t0
        info.graph_loss = loss.item()

        return x.detach()

    def validate_graph(self, x, info):
        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        x = torch.cat([x.view(-1), self.activations], 0)
        locations = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        x = self.project(x, locations, self.neuron_sources)

        loss = x.std(1).mean()

        info.graph_time = time() - t0
        info.graph_loss = loss.item()

        return x.detach()

    def forward_graph(self, x):
        assert x.shape == (1, self.in_height, self.in_width)

        x = torch.cat([x.view(-1), self.activations], 0)
        locations = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        x = self.project(x, locations, self.neuron_sources)

        return x.detach()

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

    def train_classifier(self, x, y_true, info):
        t0 = time()
        self.clf_optimizer.zero_grad()
        y_pred = self.clf(x)
        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        self.clf_optimizer.step()
        info.clf_time = time() - t0
        info.clf_loss = loss.item()
        info.clf_acc = (y_pred.argmax() == y_true).type(torch.float32).item()
        return y_pred.detach()

    def validate_classifier(self, x, y_true, info):
        t0 = time()
        y_pred = self.clf(x)
        loss = F.cross_entropy(y_pred, y_true)
        info.clf_time = time() - t0
        info.clf_loss = loss.item()
        info.clf_acc = (y_pred.argmax() == y_true).type(torch.float32).item()
        return y_pred.detach()

    def forward_classifier(self, x):
        return self.clf(x).detach()

    def train_on_tick(self, x, y):
        info = Info()
        x = self.train_graph(x, info)
        self.activations = self.train_neurons(x, info)
        y_pred = self.train_classifier(self.activations.unsqueeze(0), y, info)
        return y_pred, info

    def validate_on_tick(self, x, y):
        info = Info()
        x = self.validate_graph(x, info)
        self.activations = self.validate_neurons(x, info)
        y_pred = self.validate_classifier(self.activations.unsqueeze(0), y, info)
        return y_pred, info

    def forward_on_tick(self, x):
        x = self.forward_graph(x)
        self.activations = self.forward_neurons(x)
        return self.forward_classifier(self.activations.unsqueeze(0))
