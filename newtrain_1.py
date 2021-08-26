#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder
from dataset.cifar_simsiam import IMBALANCECIFAR10, IMBALANCECIFAR100
from dataset.cifar_pair import CIFAR10Pair, CIFAR100Pair
from simsiam.builder import simsiam_resnet32, simsiam_resnet56


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

augmentation_cifar = [
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ]

parser = argparse.ArgumentParser(description='PyTorch simsiam-cifar Training')
parser.add_argument('--data-dir', default='./data', type=str,
                    help='the diretory to save cifar100 dataset')
parser.add_argument('--dataset', '-d', type=str, default='cifar100_lt',
                    help='dataset choice')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_ratio', type=float, default=0.1, help='dataset imbalacen ratio')
parser.add_argument('--head_ratio', type=float, default=1.0, help='ratio to use on head class 1.0 for 500/5000')
parser.add_argument('--model', metavar='ARCH', default='resnet32')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save-path', default='runs/new/', type=str,
                    help='folder to save the checkpoints')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 100)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 64)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

def main():
    args = parser.parse_args()
    if 'lt' not in args.dataset:
        args.save_path = save_path = os.path.join(args.save_path,
                                                  '_'.join([
                                                      args.dataset, (str)(args.head_ratio), (str)(args.batch_size),(str)(args.epochs)
                                                  ]), 'stage1')
    else:
        args.save_path = save_path = os.path.join(args.save_path,
            '_'.join([
            args.dataset, (str)(args.imb_ratio), (str)(args.head_ratio), (str)(args.batch_size), (str)(args.epochs)
        ]), 'stage1')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_train.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dataset == 'cifar100_lt':
        train_dataset = IMBALANCECIFAR100(phase='train', imbalance_ratio=args.imb_ratio, head_ratio=args.head_ratio,
                                          root=args.data_dir, simsiam=True)
        num_classes = 100
    elif args.dataset == 'cifar10_lt':
        train_dataset = IMBALANCECIFAR10(phase='train', imbalance_ratio=args.imb_ratio, root=args.data_dir,
                                          simsiam=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(phase='train', imbalance_ratio=1.0, head_ratio=args.head_ratio,
                                         root=args.data_dir, simsiam=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(phase='train', imbalance_ratio=1.0, head_ratio=args.head_ratio,
                                         root=args.data_dir, simsiam=True)
        num_classes = 100
    else:
        warnings.warn("Wrong dataset name: ", args.dataset)

    if args.head_ratio != 1.0:
        data_path = os.path.join(args.save_path, 'dataset.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
    elif args.dataset.endswith('_lt'):
        data_path = os.path.join(args.save_path, 'dataset.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)

    if args.model == 'resnet32':
        model = simsiam_resnet32(num_classes=num_classes)
    elif args.model == 'resnet56':
        model = simsiam_resnet56(num_classes=num_classes)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True, num_classes=num_classes)
    else:
        warnings.warn("Wrong model name: ", args.model)

    init_lr = args.lr * args.batch_size / 256
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    logging.info("=> creating model '{}'".format(args.model))
    print(model)  # print model after SyncBatchNorm

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        # optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
        #                 {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.exists(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}', epoch {}".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if epoch % 50 == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch)))

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.save_path, 'checkpoint_last.pth.tar'.format(epoch)))



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (image1, image2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            image1 = image1.cuda(args.gpu, non_blocking=True)
            image2 = image2.cuda(args.gpu, non_blocking=True)
        # compute output and loss
        p1, p2, z1, z2 = model(x1=image1, x2=image2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), image1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    logging.info("Epoch: [{0}]\t"
                 "Loss {loss})\t".format(
        epoch,
        loss=losses.avg,))


def save_checkpoint(state, is_best, filename='runs/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/model_best.pth.tar')


class AverageMeter(object):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
