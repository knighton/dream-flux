from argparse import ArgumentParser
from torch.utils.data import DataLoader
  
from .dataset.mnist import load_mnist
from .model.reco_grid import RecoGrid


def parse_flags():
    a = ArgumentParser()

    # Training.
    a.add_argument('--data_loader_workers', type=int, default=2)
    a.add_argument('--epochs', type=int, default=1000)
    a.add_argument('--train_batches_per_epoch', type=int, default=50)
    a.add_argument('--val_batches_per_epoch', type=int, default=25)
    a.add_argument('--batch_size', type=int, default=10)
    a.add_argument('--cuda', type=int, default=1)
    a.add_argument('--tqdm', type=int, default=1)

    # Model dimensions.
    a.add_argument('--ticks_per_sample', type=int, default=5)
    a.add_argument('--vision_body_channels', type=int, default=32)
    a.add_argument('--vision_out_memes', type=int, default=-1)
    a.add_argument('--meme_dim', type=int, default=32)
    a.add_argument('--num_nodes', type=int, default=64)
    a.add_argument('--inputs_per_node', type=int, default=16)
    a.add_argument('--vocab_size', type=int, default=256)
    a.add_argument('--soft_clip', type=float, default=1)
    a.add_argument('--sharpness', type=float, default=2)

    return a.parse_args()


def main(f):
    in_shape = 1, 28, 28
    out_classes = 10

    if f.vision_out_memes == -1:
        f.vision_out_memes = f.vision_body_channels * 16 // f.meme_dim

    train_dataset, val_dataset = load_mnist()

    train_loader = DataLoader(
        train_dataset, batch_size=f.batch_size, shuffle=True,
        num_workers=f.data_loader_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=f.batch_size, shuffle=True,
        num_workers=f.data_loader_workers)

    model = RecoGrid(
        f.ticks_per_sample, in_shape, f.vision_body_channels, 
        f.vision_out_memes, f.meme_dim, f.num_nodes, f.inputs_per_node, f.vocab_size,
        f.soft_clip, f.sharpness, out_classes)

    if f.cuda:
        model.cuda()

    model.fit(train_loader, val_loader, f.epochs, f.train_batches_per_epoch,
              f.val_batches_per_epoch, f.cuda, f.tqdm)


if __name__ == '__main__':
    main(parse_flags())
