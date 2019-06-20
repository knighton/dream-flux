from argparse import ArgumentParser
from torch.utils.data import DataLoader
  
from .dataset.mnist import load_mnist
from .model import Model


def parse_flags():
    a = ArgumentParser()

    a.add_argument('--data_loader_workers', type=int, default=2)

    a.add_argument('--epochs', type=int, default=1000)
    a.add_argument('--train_batches_per_epoch', type=int, default=20)
    a.add_argument('--val_batches_per_epoch', type=int, default=10)
    a.add_argument('--batch_size', type=int, default=10)
    a.add_argument('--cuda', type=int, default=1)
    a.add_argument('--tqdm', type=int, default=1)

    a.add_argument('--embed_dim', type=int, default=256)
    a.add_argument('--inputs_per_neuron', type=int, default=16)
    a.add_argument('--outputs_per_neuron', type=int, default=16)
    a.add_argument('--num_neurons', type=int, default=64)
    a.add_argument('--latent_dim', type=int, default=4)
    a.add_argument('--ticks_per_sample', type=int, default=5)

    return a.parse_args()


def main(flags):
    in_height = 28
    in_width = 28
    out_classes = 10

    train_dataset, val_dataset = load_mnist()

    train_loader = DataLoader(train_dataset, batch_size=flags.batch_size,
                              shuffle=True, num_workers=flags.data_loader_workers)
    val_loader = DataLoader(val_dataset, batch_size=flags.batch_size,
                            shuffle=True, num_workers=flags.data_loader_workers)

    model = Model(in_height, in_width, out_classes, flags.embed_dim,
                  flags.inputs_per_neuron, flags.outputs_per_neuron,
                  flags.num_neurons, flags.latent_dim, flags.ticks_per_sample)

    if flags.cuda:
        model.cuda()

    model.fit(train_loader, val_loader, flags.epochs, flags.train_batches_per_epoch,
              flags.val_batches_per_epoch, flags.cuda, flags.tqdm)


if __name__ == '__main__':
    main(parse_flags())
