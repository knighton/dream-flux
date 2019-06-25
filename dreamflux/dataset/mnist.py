from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as tf


def load_mnist():
    train_dataset = MNIST(root='data/', train=True, download=True,
                          transform=tf.ToTensor())
    val_dataset = MNIST(root='data/', train=False, download=True,
                        transform=tf.ToTensor())
    return train_dataset, val_dataset
