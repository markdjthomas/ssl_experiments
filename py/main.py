# main.py
# Script for training a CNN on MNIST.
#
# author: Mark Thomas
# modifed: 2019-10-23

import argparse
import random
import warnings

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

from utils.pretraining_utils import train, validate, test_best_metrics, save_checkpoint
from torch.utils.tensorboard import SummaryWriter

# Set the global best metrics
best_metrics = (0.0, 0.0, 0.0)

# Setup the ArgumentParser
parser = argparse.ArgumentParser(description='PyTorch Training Script')

# Training arguments
parser.add_argument('--run-id', type=str, dest='run_id')
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--classes', default=10, type=int)
parser.add_argument('--print-freq', default=100, type=int, dest='print_freq')
parser.add_argument('--num-workers', default=16, type=int, dest='num_workers')

# Learning arguments
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--learning-rate', default=0.01, type=float, dest='learning_rate')
parser.add_argument('--decay-rate', default=0.1, type=float, dest='decay_rate')


def main():
    global best_metrics

    # Parse the arguments
    args = parser.parse_args()

    # Create the SummaryWriter for Tensorboard
    args.writer = SummaryWriter('./logs/tensorboard/{}'.format(args.run_id))

    # Set the RNG seegs
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. \
                       This will turn on the CUDNN deterministic setting, \
                       which can slow down your training considerably! \
                       You may see unexpected behavior when restarting \
                       from checkpoints.')

    # Print out the training setup
    print('New training run...\n')
    print('   Run ID:            {}'.format(args.run_id))
    print('   Architecture:      {}'.format(args.arch))
    print('   Batch size:        {}'.format(args.batch_size))
    print('   Learning rate:     {}'.format(args.learning_rate))
    print('   Decay rate:        {}\n'.format(args.decay_rate))

    # Create the model
    print("=> creating model...")
    device = torch.device('cuda')
    model = models.__dict__[args.arch](pretrained=False, num_classes=args.classes).to(device)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_rate, patience=10)
    cudnn.benchmark = True

    # Create the datasets and loaders
    print('=> creating the datasets and iterators')

    # Create the training dataset and loader
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    training_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = datasets.MNIST('../data', train=False, transform=transforms)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    # Save the lengths of the data loaders for Tensorboard
    args.train_loader_len = len(training_loader)
    args.validation_loader_len = len(validation_loader)

    # Train the model
    print('=> starting the training\n')
    for epoch in range(args.epochs):
        # Set the current epoch to be used by Tensorboard
        args.current_epoch = epoch

        # Take a training step
        train(training_loader, model, criterion, optimizer, epoch, device, args)

        # Evaluate on validation set and check if it is the current best
        val_loss, metrics = validate(validation_loader, model, criterion, device, args)
        best_metrics, is_best = test_best_metrics(metrics, best_metrics)

        # Take a step using the learning rate scheduler
        lr_scheduler.step(val_loss)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_metrics[0],
            'best_pre': best_metrics[1],
            'best_rec': best_metrics[2]
        }, is_best, args)

    # Close the Tensorboard writer
    args.writer.close()


if __name__ == '__main__':
    main()
