import torch
from torch import nn

from ...layer import *


class DenseBlock(nn.Sequential):
    """
    Linear/Norm/ReLU/Dropout.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            InstanceNorm(),
            nn.ReLU(),
            nn.Dropout(),
        )


class MemeWeighter(nn.Sequential):
    """
    Weight memes for combining into a summary.
    """

    def __init__(self, meme_dim):
        a = meme_dim
        b = meme_dim // 2
        c = meme_dim // 4
        d = 1
        super().__init__(
            DenseBlock(a, b),
            DenseBlock(b, c),
            nn.Linear(c, d),
            InstanceNorm(),
        )


class MemeClassifier(nn.Sequential):
    """
    Classify a single meme summarizing the population of them.
    """

    def __init__(self, meme_dim, num_classes):
        a = meme_dim
        b = meme_dim * 2
        c = meme_dim * 4
        d = num_classes
        super().__init__(
            DenseBlock(a, b),
            DenseBlock(b, c),
            nn.Linear(c, d),
        )


class Classifier(nn.Module):
    """
    Classify the sample given its resulting population of memes.
    """

    def __init__(self, meme_dim, num_classes):
        super().__init__()
        self.meme_dim = meme_dim
        self.num_classes = num_classes
        self.weight_memes = MemeWeighter(meme_dim)
        self.classify_meme = MemeClassifier(meme_dim, num_classes)

    def forward(self, x):
        """
        Given the reconstructed memes, classify the sample.

        Input
        - memes (num nodes, meme dim)

        Output
        - classes (num classes)
        """
        # Do some magic to assign weights to the meme vectors.
        weights = self.weight_memes(x).squeeze(1)

        # Combine the memes into a summary according to the weights.
        x = torch.einsum('ne,n->e', [x, weights])

        # Normalize.
        x = x - x.mean()
        x = x / (x.std() + 1e-3)

        # Classify the summary.
        return self.classify_meme(x)
