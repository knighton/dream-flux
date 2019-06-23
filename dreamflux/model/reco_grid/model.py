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

    def forward(self, x):
        x = self.input(x)
        x = self.core(x)
        return self.output(x)
