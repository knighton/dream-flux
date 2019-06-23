from time import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from ..layer.flatten import Flatten
from .model import Model


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


class Conv(Model):
    def __init__(self, in_height, in_width, out_classes):
        super().__init__()

        assert in_height == 28
        assert in_width == 28

        c = 16
        self.seq = nn.Sequential(
            # 28x28.
            nn.Conv2d(1, c, 3, 1, 0),
            nn.BatchNorm2d(c),

            # 26x26.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            nn.BatchNorm2d(c),

            # 24x24.
            nn.MaxPool2d(2),

            # 12x12.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            nn.BatchNorm2d(c),

            # 10x10.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            nn.BatchNorm2d(c),

            # 8x8.
            nn.MaxPool2d(2),

            # 4x4.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),

            # 4x4.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),

            Flatten(),

            # 16 * c.
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16 * c, 16 * c),
            nn.BatchNorm1d(16 * c),

            # 16 * c.
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16 * c, out_classes),
        )

        self.optimizer = Adam(self.seq.parameters())

    def train_on_batch(self, x, y_true):
        info = Info()
        t0 = time()
        self.optimizer.zero_grad()
        y_pred = self.seq(x)
        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred.detach(), info

    def validate_on_batch(self, x, y_true):
        info = Info()
        t0 = time()
        y_pred = self.seq(x)
        loss = F.cross_entropy(y_pred, y_true)
        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred.detach(), info

    def forward(self, x):
        return self.seq(x).detach()
