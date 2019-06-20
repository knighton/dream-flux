import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def each(xx):
    for x in xx:
        yield x


def compute_batches_per_epoch(batch_loader, max_batches_per_epoch=None):
    batches_per_epoch = len(batch_loader)
    if max_batches_per_epoch:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)
    return batches_per_epoch


def shuffle_epoch(train_loader, val_loader=None,
                  max_train_batches_per_epoch=None,
                  max_val_batches_per_epoch=None):
    train_batches_per_epoch = compute_batches_per_epoch(
        train_loader, max_train_batches_per_epoch)

    if val_loader:
        val_batches_per_epoch = compute_batches_per_epoch(
            val_loader, max_val_batches_per_epoch)
    else:
        val_batches_per_epoch = 0

    modes = [1] * train_batches_per_epoch + [0] * val_batches_per_epoch
    np.random.shuffle(modes)

    return modes


def get_next_batch(each_mode_batch, use_cuda):
    x, y_true = next(each_mode_batch)
    if use_cuda:
        x = x.cuda()
        y_true = y_true.cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    x = F.max_pool2d(x, 2)
    return x, y_true


def each_batch(train_loader, val_loader=None, max_train_batches_per_epoch=None,
               max_val_batches_per_epoch=None, use_cuda=True, use_tqdm=True):
    each_train_batch = each(train_loader)

    if val_loader:
        each_val_batch = each(val_loader)
    else:
        each_val_batch = None

    modes = shuffle_epoch(train_loader, val_loader, max_train_batches_per_epoch,
                          max_val_batches_per_epoch)
    if use_tqdm:
        modes = tqdm(modes, leave=False)

    for batch_id, is_training in enumerate(modes):
        if is_training:
            each_mode_batch = each_train_batch
        else:
            each_mode_batch = each_val_batch
        x, y_true = get_next_batch(each_mode_batch, use_cuda)
        yield batch_id, is_training, x, y_true


class Model(nn.Module):
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

    def fit_on_epoch(self, train_loader, val_loader, num_epochs,
                     train_batches_per_epoch, val_batches_per_epoch, use_cuda,
                     use_tqdm):
        batches = each_batch(train_loader, val_loader, train_batches_per_epoch,
                             val_batches_per_epoch, use_cuda, use_tqdm)
        train_infos = []
        val_infos = []
        for batch, is_training, x, y in batches:
            if is_training:
                self.train()
                y_pred, info = self.train_on_batch(x, y)
                train_infos.append(info)
                # print('T', info.clf_acc.item())
            else:
                self.eval()
                y_pred, info = self.validate_on_batch(x, y)
                val_infos.append(info)
                # print('V', info.clf_acc.item())
        return train_infos[0].mean(train_infos), val_infos[0].mean(val_infos)

    def log(self, epoch, t, v):
        # print('%6d %6.2f%% %6.2f%%' % (epoch, 100 * t.clf_acc, 100 * v.clf_acc))

        """
        x = {
            'epoch': epoch,
            'train': t.dump(),
            'val': v.dump(),
        }
        line = '%s\n' % json.dumps(x, sort_keys=True)
        print(line)
        """

        title = 'Epoch %d' % epoch
        bar = '-' * len(title)

        print()
        print('    %s' % title)
        print('    %s' % bar)
        print()
        print('Train:')
        print(t.to_text())
        print()
        print('Val:')
        print(v.to_text())
        print()

    def fit(self, train_loader, val_loader, num_epochs, train_batches_per_epoch,
            val_batches_per_epoch, use_cuda, use_tqdm):
        for epoch in range(num_epochs):
            train_info, val_info = self.fit_on_epoch(
                train_loader, val_loader, num_epochs, train_batches_per_epoch,
                val_batches_per_epoch, use_cuda, use_tqdm)
            self.log(epoch, train_info, val_info)
