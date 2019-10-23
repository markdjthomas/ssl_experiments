# pretrain_utils.py
# Utility functions for the pre-training purposes.
#
# author: Mark Thomas
# modifed: 2019-08-24

import os
import shutil
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train(loader, model, criterion, optimizer, epoch, device, args):
    """ Training function of the model.

    Args:
        loader (DataLoader): the dataloader of the training set
        model (Model): the model being trained
        criterion (nn.Loss): the loss function of the training routine
        optimizer (torch.optim): the opimization routine
        epoch (int): the current epoch
        device (torch.device): the device to put the tensors on
        args (Namespace): parsed arguments of the ArgumentParser
    """
    # Create the AverageMeters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    xent = AverageMeter()
    acc = AverageMeter()
    pre = AverageMeter()
    rec = AverageMeter()

    # Switch the model to training mode
    model.train()

    # Loop over the training DataLoader
    end = time.time()
    for i, (features, targets) in enumerate(loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move the data to the device
        features = features.to(device)
        targets = targets.to(device)

        # Compute the model output and loss
        output = model(features)
        loss = criterion(output, targets)

        # Measure the training metrics record loss
        acc_temp, pre_temp, rec_temp = get_metrics(output, targets)

        xent.update(loss.item(), features.size(0))
        acc.update(acc_temp, features.size(0))
        pre.update(pre_temp, features.size(0))
        rec.update(rec_temp, features.size(0))

        # Compute the gradient and do a SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure the elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the metrics to Tensorboard
        tboard_step = i + (args.train_loader_len * args.current_epoch)
        args.writer.add_scalar('Train/loss', xent.avg, tboard_step)
        args.writer.add_scalar('Train/accuracy', acc.avg, tboard_step)
        args.writer.add_scalar('Train/precision', pre.avg, tboard_step)
        args.writer.add_scalar('Train/recall', rec.avg, tboard_step)

        # Write the distributions to Tensorboard
        for name, param in model.named_parameters():
            if 'bn' not in name:
                args.writer.add_histogram(name, param, tboard_step)

        # Print out the metrics
        if i % args.print_freq == 0:
            print('epoch {} - step {}/{}'.format(epoch, i, len(loader)))
            print('... batch time: {:.3f} ({:.3f})'.format(batch_time.val, batch_time.avg))
            print('... loss {:.3f} ({:.3f}) | accuracy {:.3f} ({:.3f}) | '
                  'precision {:.3f} ({:.3f}) | recall {:.3f} ({:.3f})'.
                  format(xent.val, xent.avg, acc.val, acc.avg, pre.val, pre.avg, rec.val, rec.avg))


def validate(loader, model, criterion, device, args):
    """ Evaluation function for the validation and/or test set.

    Args:
        loader (DataLoader): the dataloader of the validation set
        model (Model): the model being trained
        criterion (nn.Loss): the loss function of the training routine
        device (torch.device): the device to move the tensors onto
        args (Namespace): parsed arguments of the ArgumentParser

    Returns:
        the three validation metrics: accuracy, precision, and recall
    """
    # Create the AverageMeters
    batch_time = AverageMeter()
    xent = AverageMeter()
    acc = AverageMeter()
    pre = AverageMeter()
    rec = AverageMeter()

    # Switch the model to evaluation mode
    model.eval()

    # Loop over the validation DataLoader
    with torch.no_grad():
        end = time.time()
        for i, (features, targets) in enumerate(loader):
            # Move the data to the device
            features = features.to(device)
            targets = targets.to(device)

            # Compute the model output and loss
            output = model(features)
            loss = criterion(output, targets)

            # Measure the training metrics record loss
            acc_i, pre_i, rec_i = get_metrics(output, targets)

            xent.update(loss.item(), features.size(0))
            acc.update(acc_i, features.size(0))
            pre.update(pre_i, features.size(0))
            rec.update(rec_i, features.size(0))

            # Measure the elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the metrics to Tensorboard
            tboard_step = i + (args.validation_loader_len * args.current_epoch)
            args.writer.add_scalar('Dev/loss', xent.avg, tboard_step)
            args.writer.add_scalar('Dev/accuracy', acc.avg, tboard_step)
            args.writer.add_scalar('Dev/precision', pre.avg, tboard_step)
            args.writer.add_scalar('Dev/recall', rec.avg, tboard_step)

        print('\nvalidation set'.format(i, len(loader)))
        print('... loss {:.3f} | accuracy {:.3f} | precision {:.3f} | recall {:.3f}\n'.
              format(xent.avg, acc.avg, pre.avg, rec.avg))

    return xent.avg, (acc.avg, pre.avg, rec.avg)


def validate_test_set(loader, model):
    """ Evaluation function for the validation and/or test set.

    Args:
        loader (DataLoader): the dataloader of the validation set
        model (Model): the model being trained

    Returns:
        the targets and prediction values
    """
    # Switch the model to evaluation mode
    model.eval()

    # Create empty arrays for saving to
    preds = np.array([], dtype=np.int64)
    targets = np.array([], dtype=np.int64)

    # Loop over the test DataLoader
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, (feature, target) in enumerate(loader):
                # Compute the model output and prediction
                output = model(feature).cpu().numpy()
                pred = np.argmax(output, 1)
                target = target.cpu().numpy()

                # Append the predictions to the numpy array
                preds = np.append(preds, [pred])
                targets = np.append(targets, [target])

                # Update the progress bar
                pbar.update(1)

    return preds[:-1], targets[:-1]


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    """ Saves the model as the current checkpoint, either in general
        or as the current most best, accordingly.

    Args:
        state (dict): the current state of the model
        is_best (bool): whether this is the new best run
        filename (str): the name of the checkpoint file (default is
            checkpoint.pth.tar)
    """
    model_dir = './models/{}'.format(args.run_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with open('{}/{}'.format(model_dir, filename), 'wb') as file:
        torch.save(state, file)
    if is_best:
        shutil.copyfile('{}/{}'.format(model_dir, filename), '{}/model_best.pth.tar'.format(model_dir))


def get_metrics(output, target):
    """ Calculates the metrics desired during training and validation

    Args:
        output (Tensor): the model outputs (i.e., predictions)
        target (Tensor): the target outputs (i.e., ground truth)

    Returns:
        a triple containing the accuracy, precision, and recall
    """
    with torch.no_grad():
        # Cast the tensors to numpy arrays
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        # Get the predictions from logits
        pred = np.argmax(output, 1)

        # Compute the accuracy, precision, and recall
        acc = accuracy_score(target, pred)
        pre = precision_score(target, pred, average='macro')
        rec = recall_score(target, pred, average='macro')

        return acc, pre, rec


def test_best_metrics(metrics, best_metrics):
    """ Determines if a set of metrics are better than the
        current best.

    Args:
        metrics (tuple): a triple containing the output of get_metrics

    Returns:
        the current best metrics and a boolean value if the metrics
        provided are equal to the current best
    """
    best_acc, best_pre, best_rec = best_metrics

    is_best = False
    temp_acc = metrics[0]
    temp_pre = metrics[1]
    temp_rec = metrics[2]

    best_comparison = 0.25 * best_acc + 0.75 * (2 * best_pre * best_rec / (best_pre + best_rec + 1e-8))
    temp_comparison = 0.25 * temp_acc + 0.75 * (2 * temp_pre * temp_rec / (temp_pre + temp_rec + 1e-8))

    if temp_comparison > best_comparison:
        best_acc = temp_acc
        best_pre = temp_pre
        best_rec = temp_rec
        is_best = True

    return (best_acc, best_pre, best_rec), is_best


class AverageMeter():
    """ Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Resets the varriables to zero
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Updates the variables
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
