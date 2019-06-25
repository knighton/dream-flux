from time import time
import torch
from torch.nn import functional as F
from torch.optim import Adam

from ..tick_model import TickModel
from .classifier import Classifier
from .core import Core
from .info import Info
from .model import Model
from .vision import MNIST


class RecoGrid(TickModel):
    """
    Wrapper to deal with the weird API (due to needs of other models).
    """

    def __init__(self, ticks_per_sample, in_shape, vision_body_channels,
                 vision_out_memes, meme_dim, num_nodes, inputs_per_node, vocab_size,
                 soft_clip, sharpness, num_classes):
        super().__init__(ticks_per_sample)

        input = MNIST(vision_body_channels, vision_out_memes, meme_dim)

        core = Core(meme_dim, num_nodes, vision_out_memes, inputs_per_node,
                    vocab_size, soft_clip, sharpness)

        output = Classifier(meme_dim, num_classes)

        self.model = Model(input, core, output)

        self.optimizer = Adam(self.model.parameters())

    def forward_on_tick(self, x):
        return self.model(x, None)

    def train_on_tick(self, x, y_true):
        info = Info()
        self.optimizer.zero_grad()
        t0 = time()
        y_pred = self.model(x, info)
        y_pred = y_pred.unsqueeze(0)
        loss = F.cross_entropy(y_pred, y_true)
        t1 = time()
        loss.backward()
        t2 = time()
        self.optimizer.step()
        t3 = time()
        info.time = t3 - t0
        info.forward_time = t1 - t0
        info.backward_time = t2 - t1
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred.detach(), info

    def validate_on_tick(self, x, y_true):
        info = Info()
        t0 = time()
        y_pred = self.model(x, info)
        y_pred = y_pred.unsqueeze(0)
        loss = F.cross_entropy(y_pred, y_true)
        t1 = time()
        info.time = time() - t0
        info.forward_time = t1 - t0
        info.backward_time = 0
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred.detach(), info
