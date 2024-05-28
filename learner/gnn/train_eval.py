import math
import time

import torch

""" Train and evaluation methods in training pipeline. """


def train(model, device, train_loader, criterion, optimiser):
    model.train()
    train_loss = 0

    for data in train_loader:
        data = data.to(device)
        h_true = data.y.float().to(device)

        optimiser.zero_grad(set_to_none=True)
        h_pred = model.forward(data)

        loss = criterion.forward(h_pred, h_true)
        loss.backward()
        optimiser.step()
        train_loss += loss.detach().cpu().item()

    stats = {
        "loss": train_loss / len(train_loader),
    }
    return stats


@torch.no_grad()
def evaluate(model, device, val_loader, criterion, return_true_preds=False):
    model.eval()
    val_loss = 0

    if return_true_preds:
        y_true = torch.tensor([])
        y_pred = torch.tensor([])

    for data in val_loader:
        data = data.to(device)
        h_true = data.y.float().to(device)
        h_pred = model.forward(data)

        loss = criterion.forward(h_pred, h_true)
        val_loss += loss.detach().cpu().item()

        if return_true_preds:
            y_pred = torch.cat((y_pred, h_pred.detach().cpu()))
            y_true = torch.cat((y_true, h_true.detach().cpu()))

    stats = {
        "loss": val_loss / len(val_loader),
    }
    if return_true_preds:
        stats["y_true"] = y_true
        stats["y_pred"] = y_pred

    return stats


def gnn_train(model, device, train_loader, criterion, optimiser, epoch, lr, num_epochs):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.6f')
    iter_num = len(train_loader)

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        data = data.to(device)
        h_true = data.y.float().to(device)
        current_lr = adjust_learning_rate(optimiser, lr, epoch + i / iter_num, num_epochs)
        # print(f"lr: {optimiser.param_groups[0]['lr']}")
        h_pred = model.forward(data)
        loss = criterion.forward(h_pred, h_true)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        losses.update(loss.item(), data[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses], prefix=f"Train Epoch: [{epoch + 1}]")
    progress.display(0)
    return {
        "loss": losses.avg,
    }


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter():
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, num_epochs, warmup_epochs=5.0):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = init_lr * epoch / warmup_epochs
    else:
        lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr