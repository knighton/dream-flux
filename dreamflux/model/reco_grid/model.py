from time import time
from torch import nn


class Model(nn.Module):
    """
    Model object, split into three parts.
    """

    def __init__(self, input, core, output):
        super().__init__()
        self.input = input
        self.core = core
        self.output = output

    def forward(self, x, info):
        t0 = time()
        x = self.input(x)
        t1 = time()
        x = self.core(x, info)
        t2 = time()
        x = self.output(x)
        t3 = time()
        info.input_time = t1 - t0
        info.core_time = t2 - t1
        info.output_time = t3 - t2
        return x
