# AUTOGENERATED! DO NOT EDIT! File to edit: 02_utils.ipynb (unless otherwise specified).

__all__ = ['log', 'mixup_data', 'setup_logging_streams', 'train', 'validate', 'get_param_lr_maps',
           'setup_optimizer_and_scheduler', 'checkpoint_state']


# Cell
import logging
import numpy as np
import os
import sys
import time
import torch

log = logging.getLogger('Metrics')
log.setLevel(logging.DEBUG)


# Cell
def mixup_data(x, y, criterion, alpha=1.0):
    """Compute the mixup data for batch `x, y`. Return mixed inputs, pairs of targets, and lambda."""
    batch_size = x.size()[0]
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = np.concatenate(
            [lam[:, None], 1 - lam[:, None]], 1
        ).max(1)[:, None, None, None]
        lam = torch.from_numpy(lam).float()
        if torch.cuda.is_available():
            lam = lam.cuda()
    else:
        lam = 1.
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_y = (lam * y_a) + ((1 - lam) * y_b)

    def mixup_criterion(pred):
        return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()
    def mixup_acc(pred):
        return (pred == mixed_y).sum().item()

    return mixed_x, y_a, y_b, lam, mixup_criterion, mixup_acc


# Cell
def setup_logging_streams(model, log_to_file=True, log_to_stdout=False):
    """Utility function for setting up logging handlers for `model`."""
    formatter = logging.Formatter(
        '[%(name)s][%(asctime)s][%(levelname)s]: %(message)s',
        datefmt='%m:%d:%Y:%I:%M:%S'
    )
    handlers = []
    if log_to_file:
        fh = logging.FileHandler(os.path.join(model.log_dir, 'metrics.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
        print("Logging to %s" % os.path.join(model.log_dir, 'metrics.log'))
        handlers.append(fh)

    if log_to_stdout:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
        print("Logging to STDOUT")
        handlers.append(ch)

    def clear_handlers():
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)
        print("Cleared all logging handlers")

    return clear_handlers


# Cell
def train(
    model, epoch, dataloader, criterion, optimizer, scheduler=None, mixup=False, alpha=0.4,
    logging_frequency=50
):
    """ Trains `model` on data in `dataloader` with loss `criterion` and optimization scheme
        defined by `optimizer`, with optional learning schedule defined by `scheduler`."""
    model.train()
    total, total_loss, total_correct = 0, 0., 0.

    for i, (x, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        mixed_x, y_a, y_b, lam, mixup_criterion, mixup_acc = mixup_data(
            x, y, criterion, alpha=alpha if mixup else 0.0
        )
        optimizer.zero_grad()
        output = model(mixed_x)
        prediction = torch.argmax(output, -1)
        loss = mixup_criterion(output)
        total_loss += loss.item() * len(y)
        total_correct += mixup_acc(prediction)
        total += len(y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % logging_frequency == 0 and i > 0:
            """ TODO:
            Add Tensorboard functionality here - mainly writer.add_scalar for
            overall loss, accuracy (i.e. over all epochs).
            """
            log.debug(
                "[Epoch %d, Iteration %d / %d] Training Loss: %.5f, "
                "Training Accuracy: %.5f [Projected Accuracy: %.5f]"
                % (
                    epoch,
                    i,
                    len(dataloader),
                    total_loss / total,
                    total_correct / len(dataloader.dataset),
                    (total_correct / len(dataloader.dataset)) / (i / len(dataloader))
                )
            )
    final_loss, final_acc = total_loss / total, total_correct / total
    log.info(
        "Reporting %.5f training loss, %.5f training accuracy for epoch %d." %
        (final_loss, final_acc, epoch)
    )
    return final_loss, final_acc


# Cell
def validate(
    model, epoch, dataloader, criterion, tta=False, tta_mixing=0.6, logging_frequency=50
):
    """Validates `model` on data in `dataloader` for epoch `epoch` using objective `criterion`."""
    model.eval()
    total, total_loss, total_correct = 0, 0., 0.

    for i, (x, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            if tta:
                bs, n_aug, c, h, w = x.size()
                output = model(x.view(-1, c, h, w)).view(bs, n_aug, -1)
                output = (
                    ((1 - tta_mixing) * output[:, -1, :]) + (tta_mixing * output[:, :-1, :].mean(1))
                )
            else:
                output = model(x)
            prediction = torch.argmax(output, -1)
            loss = criterion(output, y)
            total_loss += loss.item() * len(y)
            total_correct += (prediction == y).sum().item()
            total += len(y)

        if i % logging_frequency == 0 and i > 0:
            """ TODO:
            Add Tensorboard functionality here - mainly writer.add_scalar for
            overall loss, accuracy (i.e. over all epochs).
            """
            log.debug(
                "[Epoch %d, Iteration %d / %d] Validation Loss: %.5f, "
                "Validation Accuracy: %.5f [Projected Accuracy: %.5f]"
                % (
                    epoch,
                    i,
                    len(dataloader),
                    total_loss / total,
                    total_correct / len(dataloader.dataset),
                    (total_correct / len(dataloader.dataset)) / (i / len(dataloader))
                )
            )
    final_loss, final_acc = total_loss / total, total_correct / total
    log.info(
        "Reporting %.5f validation loss, %.5f validation accuracy for epoch %d." %
        (final_loss, final_acc, epoch)
    )
    return final_loss, final_acc


# Cell
def get_param_lr_maps(model, base_lr, finetune_body_factor):
    """ Output parameter LR mappings for setting up an optimizer for `model`."""
    body_parameters = [
        (param, _) for (param, _) in model.named_parameters() if param.split('.')[0] != 'out_fc'
    ]
    if type(finetune_body_factor) is float:
        print(
            "Setting up optimizer to fine-tune body with LR %.8f and head with LR %.5f" %
            (base_lr * finetune_body_factor, base_lr)
        )
        return [
            {'params': body_parameters, 'lr': base_lr * finetune_body_factor},
            {'params': model.out_fc.parameters(), 'lr': base_lr}
        ]
    else:
        lower_bound_factor, upper_bound_factor = finetune_body_factor
        print(
            "Setting up optimizer to fine-tune body with LR in range [%.8f, %.8f]"
            " and head with LR %.5f" %
            (base_lr * lower_bound_factor, base_lr * upper_bound_factor, base_lr)
        )
        lrs = np.geomspace(
            base_lr * lower_bound_factor, base_lr * upper_bound_factor,
            len(body_parameters)
        )
        param_lr_maps = [
            {'params': param, 'lr': lr} for ((_, param), lr) in
            zip(body_parameters, lrs)
        ]
        param_lr_maps.append({'params': model.out_fc.parameters(), 'lr': base_lr})
        return param_lr_maps


# Cell
def setup_optimizer_and_scheduler(param_lr_maps, base_lr, epochs, steps_per_epoch):
    """Create a PyTorch AdamW optimizer and OneCycleLR scheduler with `param_lr_maps` parameter mapping,
       with base LR `base_lr`, for training for `epochs` epochs, with `steps_per_epoch` iterations
       per epoch."""
    optimizer = torch.optim.AdamW(param_lr_maps, lr=base_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, base_lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    return optimizer, scheduler

def checkpoint_state(
    model, epoch, optimizer, scheduler, train_loss, train_acc, val_loss, val_acc
):
    """Checkpoint the state of the system, including `model` state, `optimizer` state, `scheduler`
       state, for `epoch`, saving the metrics as well."""
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch': epoch
        },
        os.path.join(model.save_dir, 'epoch_%d.pth' % epoch)
    )