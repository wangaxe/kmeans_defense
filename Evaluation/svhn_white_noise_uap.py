import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchattacks
from advertorch.defenses import MedianSmoothing2D, BitSqueezing, JPEGFilter
from cluster import Kmeans_cluster, mb_Kmeans_cluster

from preact_resnet import PreActResNet18
from vgg import VGG11, VGG16

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--fname', type=str, default='test2')
    parser.add_argument('--model',default='pr18', type=str,
                        choices=['pr18', 'vgg11', 'vgg16'])
    parser.add_argument('--iter', type=int, default=20,
                        help='The number of iterations for iterative attacks')
    parser.add_argument('--eps', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=2)
    # parser.add_argument('--attack',default='white', type=str, choices=['white', 'uap'])
    parser.add_argument('--defense', type=str, default='km',
                        choices=['km','mbkm','bs','ms','jf'])
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default='/mnt/storage0_8/torch_datasets/svhn-data')
    return parser.parse_args()

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

def main():
    args = get_args()

    logfile = './svhn/'+args.fname+'.log'
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfile,
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO)
    logger.info(args)

    _, test_loader = get_loaders(args.data_dir, args.batch_size)

    if args.model == 'pr18':
        model = PreActResNet18().cuda()
    elif args.model == 'vgg11':
        model = VGG11().cuda()
    elif args.model == 'vgg16':
        model = VGG16().cuda()
    checkpoint = torch.load('../models/SVHN_{}.pth'.format(args.model))
    model.load_state_dict(checkpoint)

    if args.defense == 'km':
        def cluster_def(in_tensor,k=args.k):
            return Kmeans_cluster(in_tensor,k)
        defense = cluster_def
    elif args.defense == 'mbkm':
        defense = mb_Kmeans_cluster
    elif args.defense == 'bs':
        bits_squeezing = BitSqueezing(bit_depth=2)
        defense = nn.Sequential(
        bits_squeezing,
        )
    elif args.defense == 'ms':
        median_filter = MedianSmoothing2D(kernel_size=3)
        defense = nn.Sequential(
        median_filter,
        )
    elif args.defense == 'jf':
        jpeg_filter = JPEGFilter(10)
        defense = nn.Sequential(
        jpeg_filter,
        )

    model.eval()
    def_correct = 0
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()
        wn = torch.zeros_like(images).cuda()
        for i in range(wn.shape[1]):
            wn[:, i, :, :].normal_(0, 8./255)
        outputs = model(images+wn)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

        def_images = defense(images+wn)
        def_outputs = model(def_images)
        _,def_predicted = torch.max(def_outputs.data, 1)
        def_correct += (def_predicted == labels.cuda()).sum()
        
    logger.info('Accuracy with white noised images:%.4f',(float(correct) / total))
    logger.info('After process, the accuracy is:%.4f',(float(def_correct) / total))

    # model.eval()
    # correct = 0
    # total = 0
    # for images, labels in adv_loader:
    #     images = images.cuda()
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels.cuda()).sum()

    # logger.info('Accuracy with Adversarial images: %.4f',(float(correct) / total))

    # model.eval()
    # correct = 0
    # total = 0
    # for images, labels in adv_loader:
    #     images = images.cuda()
    #     images = defense(images)
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels.cuda()).sum()
        
    # logger.info('Accuracy with Defenced images: %.4f',(float(correct) / total))

if __name__ == "__main__":
    main()