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
from cluster import Kmeans_cluster, mb_Kmeans_cluster

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--fname', type=str, default='test')
    parser.add_argument('--model', default='LeNet', type=str, 
                        choices=['LeNet', 'A', 'B', 'C'], help='models type')
    parser.add_argument('--attack-type', type=str, default='fgsm',
                    choices=['fgsm', 'pgd', 'rfgsm', 'deepfool'])
    parser.add_argument('--iter', type=int, default=50,
                        help='The number of iterations for iterative attacks')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--defense', type=str, default='none',
                        choices=['km','mbkm','bs','ms','jf'])
    parser.add_argument('--k', type=int, default=2)
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

    mnist_test = datasets.MNIST('../../dataset/', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)

    if args.attack_type == 'pgd':
        data_dir = "../advdata/MNIST_{}_pgd_{}-{}.pt".format(args.model,args.eps,args.iter)
        if not os.path.exists(data_dir):
            pgd_attack = torchattacks.PGD(model, eps = args.eps, alpha = args.alpha, iters=args.iter, random_start=False)
            pgd_attack.set_mode('int')
            pgd_attack.save(data_loader=test_loader, file_name=data_dir, accuracy=True)
        adv_images, adv_labels = torch.load(data_dir)

    elif args.attack_type == 'fgsm':
        data_dir = "../advdata/MNIST_{}_fgsm_{}.pt".format(args.model, args.eps)
        if not os.path.exists(data_dir):
            fgsm_attack = torchattacks.FGSM(model, eps=args.eps)
            fgsm_attack.set_mode('int')
            fgsm_attack.save(data_loader=test_loader, file_name=data_dir, accuracy=True)
        adv_images, adv_labels = torch.load(data_dir)

    elif args.attack_type == 'deepfool':
        data_dir = "../advdata/MNIST_{}_df_{}.pt".format(args.model, args.iter)
        if not os.path.exists(data_dir):
            df_attack = torchattacks.DeepFool(model, iters=args.iter)
            df_attack.set_mode('int')
            df_attack.save(data_loader=test_loader, file_name=data_dir, accuracy=True)
        adv_images, adv_labels = torch.load(data_dir)
    
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
    logger.info('Accuracy with Clean images:%.4f',(float(correct) / total))

    model.eval()
    correct = 0
    total = 0
    for images, labels in adv_loader:
        images = images.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    logger.info('Accuracy with Adversarial images: %.4f',(float(correct) / total))

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
    correct = 0
    total = 0
    for images, labels in adv_loader:
        images = images.cuda()
        images = defense(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
    logger.info('Accuracy with Defenced images: %.4f',(float(correct) / total))

if __name__ == "__main__":
    main()