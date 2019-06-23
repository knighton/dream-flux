from torch import nn

from ...layer import *


class Vision(nn.Module):
    """
    Embeds images.
    """
    pass


class MNIST(Vision):
    """
    Embeds MNIST images.
    """

    def __init__(self, body_channels, num_memes, meme_dim):
        super().__init__()

        in_shape = in_channels, in_height, in_width = 1, 28, 28

        self.in_shape = in_shape
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.body_channels = body_channels
        self.num_memes = num_memes
        self.meme_dim = meme_dim

        c = body_channels
        self.seq = nn.Sequential(
            # 28x28.
            nn.Conv2d(1, c, 3, 1, 0),
            InstanceNorm(),

            # 26x26.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            InstanceNorm(),

            # 24x24.
            nn.MaxPool2d(2),

            # 12x12.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            InstanceNorm(),

            # 10x10.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 0),
            InstanceNorm(),

            # 8x8.
            nn.MaxPool2d(2),

            # 4x4.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            InstanceNorm(),

            # 4x4.
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            InstanceNorm(),

            Flatten(),

            # 16.
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16 * c, num_memes * meme_dim),
            InstanceNorm(),
            Reshape(num_memes, meme_dim),
        )

    def forward(self, x):
        """
        Process an input image into meme vectors.

        Input
        - x (in channels, in height, in width)

        Output
        - memes (num input memes, meme dim)
        """
        assert x.shape == self.in_shape
        x = x.unsqueeze(0)
        x = self.seq(x)
        return x.squeeze(0)
