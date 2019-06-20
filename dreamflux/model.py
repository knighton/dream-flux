import numpy as np
from time import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter as P
from torch.optim import Adam
from tqdm import tqdm

from .layer import InstanceNorm, ShardedLinear


def int_interpolate(begin, end, count, eps=1e-6):
    assert 2 <= count
    fracs = np.arange(count) / (count - 1)
    xx = []
    for frac in fracs:
        x = (1 - frac) * begin + frac * end
        x = int(x + eps)
        xx.append(x)
    return xx


def sharded_binary_cross_entropy(pred, true):
    return -true * pred.log() - (1 - true) * (1 - pred).log()


class Info(object):
    fields = 'graph_time', 'graph_loss', 'd_time', 'd_pred_mean', 'd_pred_std', \
        'd_real_loss', 'd_fake_loss', 'g_time', 'g_loss', 'clf_time', \
        'clf_loss', 'clf_acc'

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
        self.d_pred_mean = None
        self.d_pred_std = None
        self.d_real_loss = None
        self.d_fake_loss = None

        self.g_time = None
        self.g_loss = None

        self.clf_time = None
        self.clf_loss = None
        self.clf_acc = None


def each(xx):
    for x in xx:
        yield x


def compute_batches_per_epoch(batch_loader, max_batches_per_epoch=None):
    batches_per_epoch = len(batch_loader)
    if max_batches_per_epoch:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)
    return batches_per_epoch


def shuffle_epoch(train_loader, val_loader=None,
                  max_train_batches_per_epoch=None,
                  max_val_batches_per_epoch=None):
    train_batches_per_epoch = compute_batches_per_epoch(
        train_loader, max_train_batches_per_epoch)

    if val_loader:
        val_batches_per_epoch = compute_batches_per_epoch(
            val_loader, max_val_batches_per_epoch)
    else:
        val_batches_per_epoch = 0

    modes = [1] * train_batches_per_epoch + [0] * val_batches_per_epoch
    np.random.shuffle(modes)

    return modes


def get_next_batch(each_mode_batch, use_cuda):
    x, y_true = next(each_mode_batch)
    if use_cuda:
        x = x.cuda()
        y_true = y_true.cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    return x, y_true


def each_batch(train_loader, val_loader=None, max_train_batches_per_epoch=None,
               max_val_batches_per_epoch=None, use_cuda=True, use_tqdm=True):
    each_train_batch = each(train_loader)

    if val_loader:
        each_val_batch = each(val_loader)
    else:
        each_val_batch = None

    modes = shuffle_epoch(train_loader, val_loader, max_train_batches_per_epoch,
                          max_val_batches_per_epoch)
    if use_tqdm:
        modes = tqdm(modes, leave=False)

    for batch_id, is_training in enumerate(modes):
        if is_training:
            each_mode_batch = each_train_batch
        else:
            each_mode_batch = each_val_batch
        x, y_true = get_next_batch(each_mode_batch, use_cuda)
        yield batch_id, is_training, x, y_true


class BaseModel(nn.Module):
    def __init__(self, in_height, in_width, out_classes, embed_dim,
                 inputs_per_neuron, outputs_per_neuron, num_neurons, latent_dim,
                 ticks_per_sample, graph_optimizer, g, g_optimizer, d, d_optimizer,
                 clf, clf_optimizer):
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

    @classmethod
    def project(cls, values, from_locations, to_locations):
        x = torch.einsum('esi,eso->sio', [from_locations, to_locations])
        x = x - x.mean(1, keepdim=True)
        x = x / x.std(1, keepdim=True)
        x = x.exp()
        heatmap = x / x.sum(1, keepdim=True)
        return torch.einsum('i,sio->so', [values, heatmap])

    def train_graph(self, x, info):
        t0 = time()

        assert x.shape == (1, self.in_height, self.in_width)

        self.graph_optimizer.zero_grad()

        x = torch.cat([x.view(-1), self.activations], 0)
        locations = torch.cat([self.input_sinks, self.neuron_sinks], 2)

        x = self.project(x, locations, self.neuron_sources)

        loss = x.std(1).mean()
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
        dr_loss = sharded_binary_cross_entropy(is_real, target).mean()
        dr_loss.backward()

        ret = is_real.detach()

        latent = torch.randn(1, self.latent_dim, self.num_neurons, device=x.device)
        fake = self.g(latent)
        target.fill_(0)
        is_real = self.d(fake.detach())
        df_loss = sharded_binary_cross_entropy(is_real, target).mean()
        df_loss.backward()

        self.d_optimizer.step()

        t1 = time()

        self.g_optimizer.zero_grad()
        target.fill_(1)
        is_real = self.d(fake)
        g_loss = sharded_binary_cross_entropy(is_real, target).mean()
        g_loss.backward()
        self.g_optimizer.step()

        t2 = time()
        info.d_time = t1 - t0
        info.d_pred_mean = ret.mean()
        info.d_pred_std = ret.std()
        info.d_real_loss = dr_loss.item()
        info.d_fake_loss = df_loss.item()
        info.g_time = t2 - t1
        info.g_loss = g_loss.item()

        return ret.squeeze()

    def validate_neurons(self, x, info):
        t0 = time()

        real = x.unsqueeze(0)
        target = torch.full((1, self.num_neurons), 1, device=x.device)
        is_real = self.d(real)
        dr_loss = sharded_binary_cross_entropy(is_real, target).mean()

        ret = is_real.detach()

        latent = torch.randn(1, self.latent_dim, self.num_neurons, device=x.device)
        fake = self.g(latent)
        target.fill_(0)
        is_real = self.d(fake.detach())
        df_loss = sharded_binary_cross_entropy(is_real, target).mean()

        t1 = time()

        target.fill_(1)
        is_real = self.d(fake)
        g_loss = sharded_binary_cross_entropy(is_real, target).mean()

        t2 = time()
        info.d_time = t1 - t0
        info.d_pred_mean = ret.mean()
        info.d_pred_std = ret.std()
        info.d_real_loss = dr_loss.item()
        info.d_fake_loss = df_loss.item()
        info.g_time = t2 - t1
        info.g_loss = g_loss.item()

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
        info.clf_acc = (y_pred.argmax() == y_true).type(torch.float32)
        return y_pred.detach()

    def validate_classifier(self, x, y_true, info):
        t0 = time()
        y_pred = self.clf(x)
        loss = F.cross_entropy(y_pred, y_true)
        info.clf_time = time() - t0
        info.clf_loss = loss.item()
        info.clf_acc = (y_pred.argmax() == y_true).type(torch.float32)
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

    def train_on_sample(self, x, y):
        yy_pred = []
        infos = []
        for i in range(self.ticks_per_sample):
            y_pred, info = self.train_on_tick(x, y)
            yy_pred.append(y_pred)
            infos.append(info)
        return torch.stack(yy_pred, 0).mean(0), Info.mean(infos)

    def validate_on_sample(self, x, y):
        yy_pred = []
        infos = []
        for i in range(self.ticks_per_sample):
            y_pred, info = self.validate_on_tick(x, y)
            yy_pred.append(y_pred)
            infos.append(info)
        return torch.stack(yy_pred, 0).mean(0), Info.mean(infos)

    def forward_on_sample(self, x):
        yy = []
        for i in range(self.ticks_per_sample):
            y = self.forward_on_tick(x)
            yy.append(y)
        return torch.stack(yy, 0).mean(0)

    def train_on_batch(self, xx, yy):
        yy_pred = []
        infos = []
        for x, y in zip(xx, yy):
            y = torch.LongTensor([y])
            if x.is_cuda:
                y = y.cuda()
            y_pred, info = self.train_on_sample(x, y)
            yy_pred.append(y_pred)
            infos.append(info)
        return torch.stack(yy_pred, 0), Info.mean(infos)

    def validate_on_batch(self, xx, yy):
        yy_pred = []
        infos = []
        for x, y in zip(xx, yy):
            y = torch.LongTensor([y])
            if x.is_cuda:
                y = y.cuda()
            y_pred, info = self.validate_on_sample(x, y)
            yy_pred.append(y_pred)
            infos.append(info)
        return torch.stack(yy_pred, 0), Info.mean(infos)

    def forward(self, xx):
        yy = []
        for x in xx:
            y = self.forward_on_sample(x)
            yy.append(y)
        return torch.stack(yy, 0).mean(0)

    def fit_on_epoch(self, train_loader, val_loader, num_epochs,
                     train_batches_per_epoch, val_batches_per_epoch, use_cuda,
                     use_tqdm):
        batches = each_batch(train_loader, val_loader, train_batches_per_epoch,
                             val_batches_per_epoch, use_cuda, use_tqdm)
        train_infos = []
        val_infos = []
        for batch, is_training, x, y in batches:
            if is_training:
                self.train()
                y_pred, info = self.train_on_batch(x, y)
                train_infos.append(info)
                # print('T', info.clf_acc.item())
            else:
                self.eval()
                y_pred, info = self.validate_on_batch(x, y)
                val_infos.append(info)
                # print('V', info.clf_acc.item())
        return Info.mean(train_infos), Info.mean(val_infos)

    def log(self, epoch, t, v):
        t = 100 * float(t.clf_acc)
        v = 100 * float(v.clf_acc)
        print('%6d %6.2f%% %6.2f%%' % (epoch, t, v))

    def fit(self, train_loader, val_loader, num_epochs, train_batches_per_epoch,
            val_batches_per_epoch, use_cuda, use_tqdm):
        for epoch in range(num_epochs):
            train_info, val_info = self.fit_on_epoch(
                train_loader, val_loader, num_epochs, train_batches_per_epoch,
                val_batches_per_epoch, use_cuda, use_tqdm)
            self.log(epoch, train_info, val_info)


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


class Model(BaseModel):
    def __init__(self, in_height, in_width, out_classes, embed_dim,
                 inputs_per_neuron, outputs_per_neuron, num_neurons, latent_dim,
                 ticks_per_sample):
        graph_optimizer = Adam

        a, b, c, d = int_interpolate(latent_dim, inputs_per_neuron, 4)
        g_model = nn.Sequential(
            GANBlock(num_neurons, a, b),
            GANBlock(num_neurons, b, c),
            ShardedLinear(num_neurons, c, d),
            nn.BatchNorm1d(d),
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
