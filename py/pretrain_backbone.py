# pretrain_backbone.py
# Script for pre-training the ResNet backbone used for R-CNN.
#
# author: Mark Thomas
# modifed: 2019-10-10

import argparse
import random
import pickle
import warnings

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import models
from utils.data_utils import StreamingDataset
from utils.pretraining_utils import train, validate, test_best_metrics, save_checkpoint
from torch.utils.tensorboard import SummaryWriter

# Set the global best metrics and recieve the available models
best_metrics = (0.0, 0.0, 0.0)
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# Setup the ArgumentParser
parser = argparse.ArgumentParser(description='PyTorch MammalNet Backbone Pre-training')

# Data arguments
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--classes', default=3, type=int)
parser.add_argument('--seconds', default=5, type=int)
parser.add_argument('--fmax', default=1024, type=int)
parser.add_argument('--foreground', default=0, type=int)
parser.add_argument('--ambient-prob', default=0, type=float, dest='ambient_prob')

# Training arguments
parser.add_argument('--run-id', type=str, dest='run_id')
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--split', default=0, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--print-freq', default=100, type=int, dest='print_freq')
parser.add_argument('--num-workers', default=24, type=int, dest='num_workers')

# Learning arguments
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--frozen-batch-norm', default=0, type=int, dest='frozen_batch_norm')
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
    print('   Frozen batchnorm:  {}'.format(bool(args.frozen_batch_norm)))
    print('   Data split:        {}'.format(args.split))
    print('   Ambient prob:      {}'.format(args.ambient_prob))
    print('   Seconds:           {}'.format(args.seconds))
    print('   Max frequency:     {}'.format(args.fmax))
    print('   Batch size:        {}'.format(args.batch_size))
    print('   Learning rate:     {}'.format(args.learning_rate))
    print('   Decay rate:        {}\n'.format(args.decay_rate))
    
    # Create the model
    print("=> creating model...")
    device = torch.device('cuda')
    
    if args.arch.startswith('resnet') and args.frozen_batch_norm: 
        model = models.__dict__[args.arch](pretrained=False, args=args, num_classes=args.classes, norm_layer=torchvision.ops.misc.FrozenBatchNorm2d).to(device)
    elif args.arch.startswith('shuffle'):
        assert args.channels == 1, "ShuffleNetv2 can only be trained on single-channel tensors" 
        model = models.__dict__[args.arch](pretrained=False, num_classes=args.classes).to(device)
    else:
        model = models.__dict__[args.arch](pretrained=False, args=args, num_classes=args.classes).to(device)
    
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_rate, patience=5)
    cudnn.benchmark = True

    # Create the datasets and loaders
    print('=> creating the datasets and iterators')

    # Specify the class_lookup dictionary
    if args.ambient_prob > 0:
        class_lookup = {'AB': 0, 'BW': 1, 'FW': 2, 'SW': 3}
    else:
        class_lookup = {'BW': 0, 'FW': 1, 'SW': 2}

    # Create the training dataset and loader
    with open('./data/splits/training_split_{}_BW_FW_SW.pickle'.format(args.split), 'rb') as f:
        training_file_dict = pickle.load(f)

    training_identifiers = list(training_file_dict.keys())

    training_dataset = StreamingDataset(training_identifiers, training_file_dict, class_lookup,
                                        seconds=args.seconds, foreground=args.foreground, fmax=args.fmax,
                                        pretraining=True, ambient_prob=args.ambient_prob)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # Create the validation dataset and loader
    with open('./data/splits/validation_split_{}_BW_FW_SW.pickle'.format(args.split), 'rb') as f:
        validation_file_dict = pickle.load(f)

    validation_identifiers = list(validation_file_dict.keys())

    validation_dataset = StreamingDataset(validation_identifiers, validation_file_dict, class_lookup,
                                          seconds=args.seconds, foreground=args.foreground, fmax=args.fmax,
                                          pretraining=True, ambient_prob=args.ambient_prob)

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)

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
    writer.close()


if __name__ == '__main__':
    main()
