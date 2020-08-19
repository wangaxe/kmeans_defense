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

from mnist_net import Le_Net, classifier_A, classifier_B, classifier_C
from cluster import Kmeans_cluster

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--fname', type=str, default='test')
    parser.add_argument('--model', default='LeNet', type=str, 
                        choices=['LeNet', 'A', 'B', 'C'], help='models type')
    parser.add_argument('--iter', type=int, default=50,
                        help='The number of iterations for iterative attacks')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--defense', type=str, default='km',
                        choices=['km','bs','ms','jf'])
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default='../../datasets/')
    return parser.parse_args()

def main():
    args = get_args()

    logfile = './mnist/'+args.fname+'.log'
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfile,
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO)
    logger.info(args)
    if not os.path.exists('../advdata'):
        os.mkdir('../advdata')

    if args.model == 'A':
        model = classifier_A().cuda()
        checkpoint = torch.load('../models/MNIST_A.pth')
    elif args.model == 'B':
        model = classifier_B().cuda()
        checkpoint = torch.load('../models/MNIST_B.pth')
    elif args.model == 'C':
        model = classifier_C().cuda()
        checkpoint = torch.load('../models/MNIST_C.pth')
    elif args.model == 'LeNet':
        model = Le_Net().cuda()
        checkpoint = torch.load('../models/MNIST_LeNet.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    mnist_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)

    if args.defense == 'km':
        def cluster_def(in_tensor,k=args.k):
            return Kmeans_cluster(in_tensor,k)
        defense = cluster_def
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
        # wn = torch.zeros_like(images).normal_(0, args.eps).cuda()
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

if __name__ == "__main__":
    main()