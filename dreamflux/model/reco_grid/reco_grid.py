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
        return self.core(x)

    def train_on_tick(self, x, y_true):
        info = Info()
        t0 = time()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
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
        y_pred = self.model(x)
        y_pred = y_pred.unsqueeze(0)
        loss = F.cross_entropy(y_pred, y_true)
        info.time = time() - t0
        info.loss = loss.item()
        info.acc = (y_pred.max(1)[1] == y_true).type(torch.float).mean().item()
        return y_pred.detach(), info
