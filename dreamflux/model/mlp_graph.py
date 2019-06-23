from time import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter as P
from torch.optim import Adam

from ..layer.instance_norm import InstanceNorm
from ..layer.sharded_linear import ShardedLinear
from .tick_model import TickModel


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


class ShardedLinearBlock(nn.Sequential):
    def __init__(self, num_shards, shard_in_dim, shard_out_dim):
        super().__init__(
            ShardedLinear(num_shards, shard_in_dim, shard_out_dim),
            InstanceNorm(),
            nn.ReLU(),
            nn.Dropout(),
        )


class MLPGraph(TickModel):
    def __init__(self, in_height, in_width, out_classes, embed_dim,
                 inputs_per_neuron, outputs_per_neuron, num_neurons, latent_dim,
                 ticks_per_sample):
        super().__init__()

        in_dim = in_height * in_width

        self.in_height = in_height
        self.in_width = in_width
        self.in_dim = in_dim
        self.out_classes = out_classes
        self.embed_dim = embed_dim
        self.inputs_per_neuron = inputs_per_neuron
        self.outputs_per_neuron = outputs_per_neuron
        self.num_neurons = num_neurons
        self.ticks_per_sample = ticks_per_sample

        x = torch.randn(embed_dim, outputs_per_neuron, in_dim)
        self.input_dests = P(x)

        x = torch.randn(embed_dim, inputs_per_neuron, num_neurons)
        self.neuron_sources = P(x)

        # (batch size, inputs per neuron, num neurons)
        #     ::
        # (batch size, outputs per neuron, num neurons)
        mids_per_neuron = (inputs_per_neuron + outputs_per_neuron) // 2
        self.predict = nn.Sequential(
            ShardedLinearBlock(num_neurons, inputs_per_neuron, mids_per_neuron),
            ShardedLinearBlock(num_neurons, mids_per_neuron, outputs_per_neuron),
            ShardedLinear(num_neurons, outputs_per_neuron, outputs_per_neuron),
        )

        self.register_buffer('pred_out', torch.rand(outputs_per_neuron, num_neurons))

        x = torch.randn(embed_dim, outputs_per_neuron, num_neurons)
        self.neuron_dests = P(x)

        x = torch.randn(embed_dim, inputs_per_neuron, out_classes)
        self.output_sources = P(x)

        parameters = [self.input_dests, self.neuron_sources] + \
            list(self.predict.parameters()) + \
            [self.neuron_dests, self.output_sources]
        self.optimizer = Adam(parameters)

    def project(self, values, from_locations, to_locations):
        # Relate the source and destination vectors.
        from_ = from_locations.view(self.embed_dim, -1)
        to = to_locations.view(self.embed_dim, -1)
        io = torch.einsum('ei,eo->io', [from_, to])
        assert not torch.isnan(from_).any()
        assert not torch.isnan(to).any()
        assert not torch.isnan(io).any()

        # Normalize scores to a bell curve around zero of sources per destination.
        io = io - io.mean(1, keepdim=True)
        io = io / (io.std(1, keepdim=True) + 1e-3)
        assert not torch.isnan(io).any()

        # Apply a smooth cutoff to the scores (-3, 3).
        io = (io / 3).tanh() * 3

        # Translate that to weights (0.05, 20).
        io = io.exp()
        assert not torch.isnan(io).any()

        # Make those weights fractions (of sources per destination).
        io = io / (io.sum(1, keepdim=True) + 1e-3)
        assert not torch.isnan(io).any()

        # Unpack shapes.
        io = io.view(self.outputs_per_neuron, self.in_dim + self.num_neurons,
                     self.inputs_per_neuron, -1)

        # Do the projection.
        x = torch.einsum('oa,oaib->ib', [values, io])

        # Put in the range (0, 1).
        return x.sigmoid()

    def train_on_tick(self, x, y_true):
        info = Info()

        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        x = x.view(1, -1)
        x = x.repeat(self.outputs_per_neuron, 1)
        x = F.dropout(x, 0.75)

        act_at_dests = torch.cat([x, self.pred_out], 1)

        dests = torch.cat([self.input_dests, self.neuron_dests], 2)

        y_pred = self.project(act_at_dests, dests, self.output_sources)
        y_pred = y_pred.mean(0, keepdim=True)
        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pred_in = self.project(act_at_dests.detach(), dests.detach(),
                               self.neuron_sources.detach())
        self.pred_out = self.predict(pred_in.unsqueeze(0)).squeeze(0)

        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred[0].argmax() == y_true).type(torch.float).item()

        return y_pred, info

    def validate_on_tick(self, x, y_true):
        info = Info()

        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        x = x.view(1, -1)
        x = x.repeat(self.outputs_per_neuron, 1)
        x = F.dropout(x, 0.75)
        act_at_dests = torch.cat([x, self.pred_out], 1)
        dests = torch.cat([self.input_dests, self.neuron_dests], 2)

        y_pred = self.project(act_at_dests, dests, self.output_sources)
        y_pred = y_pred.mean(0, keepdim=True)
        loss = F.cross_entropy(y_pred, y_true)

        pred_in = self.project(act_at_dests.detach(), dests.detach(),
                               self.neuron_sources.detach())
        self.pred_out = self.predict(pred_in.unsqueeze(0)).squeeze(0)

        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred[0].argmax() == y_true).type(torch.float).item()

        return y_pred, info

    def forward_on_tick(self, x):
        assert x.shape == (1, self.in_height, self.in_width)

        x = x.view(1, -1)
        x = x.repeat(self.outputs_per_neuron, 1)
        x = F.dropout(x, 0.75)
        act_at_dests = torch.cat([x, self.pred_out], 1)
        dests = torch.cat([self.input_dests, self.neuron_dests], 2)

        y_pred = self.project(act_at_dests, dests, self.output_sources)
        y_pred = y_pred.mean(0, keepdim=True)

        pred_in = self.project(act_at_dests.detach(), dests.detach(),
                               self.neuron_sources.detach())
        self.pred_out = self.predict(pred_in.unsqueeze(0)).squeeze(0)

        return y_pred
