import argparse
import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from preact_resnet import PreActResNet18
from vgg import VGG11, VGG16


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([transforms.ToTensor(),])
    test_transform = transforms.Compose([transforms.ToTensor(),])
    
    num_workers = 2
    
    train_dataset = datasets.SVHN(
        dir_, split='train', transform=train_transform, download=True)
    test_dataset = datasets.SVHN(
        dir_, split='test', transform=test_transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,)
    return train_loader, test_loader

def clean_evaluate(test_loader, model):
    total_loss = 0
    total_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            total_loss += loss.item() * y.size(0)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return total_loss/n, total_acc/n


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../svhndata/', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'flat'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--fname', default='train', type=str)
    parser.add_argument('--model',default='pr18', type=str, choices=['pr18', 'vgg11', 'vgg16'])
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively ad just the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()

def main():
    args = get_args()

    logfile = args.fname+'.log'
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfile,
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO)

    logger.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    if args.model == 'pr18':
        model = PreActResNet18().cuda()
    elif args.model == 'vgg11':
        model = VGG11().cuda()
    elif args.model == 'vgg16':
        model = VGG16().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'flat':
        lr_lamdbda = lambda t: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lamdbda)
        
    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    torch.save(model.state_dict(), '../models/SVHN_{}.pth'.format(args.model))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
        

    # Evaluation
    if args.model == 'pr18':
        model_test = PreActResNet18().cuda()
    elif args.model == 'vgg11':
        model_test = VGG11().cuda()
    elif args.model == 'vgg16':
        model_test = VGG16().cuda()
        
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()
    test_loss, test_acc = clean_evaluate(test_loader, model_test)

    logger.info('Test Loss \t Test Acc')
    logger.info('{:.4f} \t\t {:.4f} '.format(test_loss, test_acc))

if __name__ == "__main__":
    main()
