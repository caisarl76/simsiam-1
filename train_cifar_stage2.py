#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import logging
import json
import pickle5 as pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataset.cifar_simsiam import IMBALANCECIFAR10, IMBALANCECIFAR100
from simsiam.resnet_cifar import resnet32, resnet56

train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-dir', default='./data', type=str,
                    help='the diretory to save cifar100 dataset')
parser.add_argument('--dataset', '-d', type=str, default='cifar100_lt',
                    choices=['cifar10', 'cifar100', 'cifar10_lt', 'cifar10_part', 'cifar100_part', 'cifar100_lt', 'imagenet_lt'],
                    help='dataset choice')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_ratio', type=float, default=0.1, help='dataset imbalacen ratio')
parser.add_argument('--head_ratio', type=float, default=1.0, help='ratio to use on head class 1.0 for 500/5000')
parser.add_argument('--model', metavar='ARCH', default='resnet32')
parser.add_argument('--loss_type', type=str, default='CE')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--lr_interval', default=[150, 180], type=int, nargs='+')
parser.add_argument('--lr_sche', default='step', type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--supervised', default=0, type=int,
                    help='choose weather run sup learning')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')

best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.supervised == 1:
        print('train with supervised')
        stage2_fol = 'super'
    else:
        print('train with supervised')
        stage2_fol = 'unsup'
    args.save_path = save_path = os.path.join(args.pretrained.split('checkpoint')[0].replace('stage1', stage2_fol), args.dataset,
                                              '_'.join([args.loss_type, (str)(args.epochs), (str)(args.lr), args.lr_sche]))
    print(args.save_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    args.logger_file = os.path.join(save_path, 'log_train.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    logging.info('start training stage2: {}'.format(stage2_fol))

    with open(os.path.join(save_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.model))
    logging.info("=> creating model '{}'".format(args.model))
    writer = SummaryWriter(os.path.join(args.save_path, 'logs'))

    cls_num_list = None
    if args.dataset.endswith('_part'):
        args.dataset = args.dataset.split('_part')[0]
        args.head_ratio = 0.4

    if args.dataset == 'cifar100_lt':
        train_dataset = IMBALANCECIFAR100(phase='train', imbalance_ratio=args.imb_ratio, head_ratio=args.head_ratio,
                                          root=args.data_dir, simsiam=False)
        cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=val_transform)
        num_classes = 100
    elif args.dataset == 'cifar10_lt':
        train_dataset = IMBALANCECIFAR10(phase='train', imbalance_ratio=args.imb_ratio, root=args.data_dir,
                                          simsiam=False)
        cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=val_transform)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(phase='train', imbalance_ratio=1.0, head_ratio=args.head_ratio,
                                         root=args.data_dir)
        cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=val_transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(phase='train', imbalance_ratio=1.0, head_ratio=args.head_ratio,
                                         root=args.data_dir)
        cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=val_transform)
        num_classes = 100
    else:
        warnings.warn("Wrong dataset name: ", args.dataset)

    if args.head_ratio != 1.0:
        data_path = os.path.join(args.pretrained.split('checkpoint')[0], 'dataset.pkl')
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                train_dataset = pickle.load(f)
                train_dataset.simsiam = False
            print(len(train_dataset))
            print(len(val_dataset))
        elif 'scratch' in args.pretrained:
            print('no dataset found! ')
    elif '_lt' in args.dataset:
        data_path = os.path.join(args.pretrained.split('checkpoint')[0], 'dataset.pkl')
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                train_dataset = pickle.load(f)
                train_dataset.simsiam = False
            print(len(train_dataset))
            print(len(val_dataset))
        elif 'scratch' in args.pretrained:
            print('no dataset found! ')

    if args.model == 'resnet32':
        model = resnet32(num_classes=num_classes)
    elif args.model == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif args.model == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=False, num_classes=num_classes)
    else:
        warnings.warn("Wrong model name: ", args.model)

    # freeze all layers but the last linear
    if not args.supervised == 1:
        for name, param in model.named_parameters():
            if name not in ['linear.weight', 'linear.bias', 'fc.weight', 'fc.bias']:
                param.requires_grad = False
    # init the linear layer
    if args.model == 'resnet18':
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    else:
        model.linear.weight.data.normal_(mean=0.0, std=0.01)
        model.linear.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            logging.info("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                # if k.startswith('module.encoder') and not k.startswith('module.encoder.linear'):
                if args.model =='resnet18':
                    if k.startswith('encoder') and not k.startswith('encoder.fc'):
                        state_dict[k[len("encoder."):]] = state_dict[k]
                else:
                    if k.startswith('encoder') and not k.startswith('encoder.linear'):
                        # remove prefix
                        # state_dict[k[len("module.encoder."):]] = state_dict[k]
                        state_dict[k[len("encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            if args.model == 'resnet18':
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            else:
                assert set(msg.missing_keys) == {"linear.weight", "linear.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.model.startswith('alexnet') or args.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss_type == 'balacned':
        from losses.BalancedSoftmaxLoss import create_loss
        criterion = create_loss(cls_num_list=cls_num_list)

    if args.supervised:
        parameters = model.parameters()
    else:
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.")
        # from apex.parallel.LARC import LARC
        # optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    cudnn.benchmark = True

    # Data loading code
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1, loss = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if epoch % 10 == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, file_dir=args.save_path)
                # if epoch == args.start_epoch and not args.supervised:
                #     sanity_check(model.state_dict(), args.pretrained)

        logging.info("Epoch: [{0}]\t"
                     "Loss {loss})\t"
                     "Prec@1 {top1:.3f})\t".format(
            epoch,
            loss=loss,
            top1=acc1)
        )
        writer.add_scalar('val loss', loss, epoch)
        writer.add_scalar('val acc', acc1, epoch)
    logging.info("Best Prec@1 {top1:.3f}".format(top1=best_acc1))
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if args.supervised == 1:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    writer.add_scalar('train loss', losses.avg, epoch)
    writer.add_scalar('train acc', top1.avg, epoch)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, file_dir='runs'):
    filename = os.path.join(file_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(file_dir,'model_best.pth.tar'))


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'linear.weight' in k or 'linear.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
    if args.lr_sche == 'cos':
        lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = init_lr * epoch / 5
        elif epoch > args.lr_interval[1]:
            lr = init_lr * 0.01
        elif epoch > args.lr_interval[0]:
            lr = init_lr * 0.1
        else:
            lr = init_lr


    """Decay the learning rate based on schedule"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
