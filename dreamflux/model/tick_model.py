import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .model import Model


class TickModel(Model):
    def __init__(self, ticks_per_sample):
        super().__init__()
        self.ticks_per_sample = ticks_per_sample

    def train_on_tick(self, x, y):
        raise NotImplementedError

    def validate_on_tick(self, x, y):
        raise NotImplementedError

    def forward_on_tick(self, x):
        raise NotImplementedError

    def train_on_sample(self, x, y):
        yy_pred = []
        infos = []
        for i in range(self.ticks_per_sample):
            y_pred, info = self.train_on_tick(x, y)
            yy_pred.append(y_pred)
            infos.append(info)
        return torch.stack(yy_pred, 0).mean(0), infos[0].mean(infos)

    def validate_on_sample(self, x, y):
        yy_pred = []
        infos = []
        for i in range(self.ticks_per_sample):
            y_pred, info = self.validate_on_tick(x, y)
            yy_pred.append(y_pred)
            infos.append(info)
        return torch.stack(yy_pred, 0).mean(0), infos[0].mean(infos)

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
        return torch.stack(yy_pred, 0), infos[0].mean(infos)

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
        return torch.stack(yy_pred, 0), infos[0].mean(infos)

    def forward(self, xx):
        yy = []
        for x in xx:
            y = self.forward_on_sample(x)
            yy.append(y)
        return torch.stack(yy, 0).mean(0)
