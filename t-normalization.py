import os, sys, shutil
import torch
import argparse
import logging
import json
import torchvision
import torchvision.transforms as transforms
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from simsiam.resnet_cifar import resnet32, resnet56
from dataset.cifar_simsiam import IMBALANCECIFAR10, IMBALANCECIFAR100

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-dir', default='./data', type=str,
                    help='the diretory to save cifar100 dataset')
parser.add_argument('--dataset', '-d', type=str, default='cifar100_lt',
                    help='dataset choice')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_ratio', type=float, default=0.1, help='dataset imbalacen ratio')
parser.add_argument('--model', metavar='ARCH', default='resnet32')
parser.add_argument('--loss_type', type=str, default='CE')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--tau', default=None, type=float, help='value of tau to use on stage2')
parser.add_argument('--module', action='store_true')

best_acc1 = 0



def main():
    args = parser.parse_args()
    args.save_path = save_path = args.pretrained.split('model_best')[0]
    # print(args.save_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    args.logger_file = os.path.join(save_path, 'log_train.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    with open(os.path.join(save_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        # logging.info("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")
    else:
        print('wrong path', args.pretrained)
        return

    if args.tau:
        weight_key = 'module.linear.weight' if args.module else 'linear.weight'
        bias_key = 'module.linear.bias' if args.module else 'linear.bias'
        weights = checkpoint['state_dict'][weight_key].cpu()
        bias = checkpoint['state_dict'][bias_key].cpu()
        ws = pnorm(weights, p=args.tau)
        bs = bias * 0
        checkpoint['state_dict'][weight_key] = ws
        checkpoint['state_dict'][bias_key] = bs

    # torch.save(checkpoint, os.path.join(save_path, 't_normed_model.pth.tar'))
    global num_classes
    global head_class_idx
    global med_class_idx
    global tail_class_idx

    if args.dataset.startswith('cifar100'):
        val_dataset = IMBALANCECIFAR100(phase='test', imbalance_ratio=1.0, root=args.data_dir, simsiam=False)
        num_classes = 100
        head_class_idx = [0, 36]
        med_class_idx = [36, 71]
        tail_class_idx = [71, 100]
    elif args.dataset.startswith('cifar10'):
        val_dataset = IMBALANCECIFAR10(phase='test', imbalance_ratio=1.0, root=args.data_dir, simsiam=False)
        num_classes = 10
        head_class_idx = [0, 3]
        med_class_idx = [3, 7]
        tail_class_idx = [7, 10]

    # print(head_class_idx, med_class_idx, tail_class_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.model == 'resnet32':
        model = resnet32(num_classes=num_classes)
    elif args.model == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif args.model == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=False, num_classes=num_classes)
    else:
        warnings.warn("Wrong model name: ", args.model)

    state_dict = checkpoint['state_dict']
    # for k in list(state_dict.keys()):
    #     if k.startswith('module'):
    #         # remove prefix
    #         # state_dict[k[len("module.encoder."):]] = state_dict[k]
    #         state_dict[k[len("module."):]] = state_dict[k]
    #     # delete renamed or unused k
    #     del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    validate(args, val_loader, model, criterion)


def validate(args, val_loader, model, criterion):
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    class_num = torch.zeros(num_classes).cuda()
    correct = torch.zeros(num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    with torch.no_grad():
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

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, num_classes)
            predict_one_hot = F.one_hot(predicted, num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

        acc_classes = correct / class_num
        head_acc = acc_classes[head_class_idx[0]:head_class_idx[1]].mean() * 100
        med_acc = acc_classes[med_class_idx[0]:med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[tail_class_idx[0]:tail_class_idx[1]].mean() * 100

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} HAcc {head_acc:.3f} MAcc {med_acc:.3f} TAcc {tail_acc:.3f}.'
              .format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))


def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws


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
