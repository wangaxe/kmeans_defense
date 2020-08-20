import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
# from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import argparse
# from model_mnist import Basic_CNN,modelA,modelB,modelC
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import logging
import os
import time
import datetime
import random
import torchvision.models as models

# import attack_model
# from utils import *
# from models import *
# from vgg import VGG
# import torch.backends.cudnn as cudnn
# import pretrainedmodels
# from dictances import bhattacharyya_coefficient
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from cluster import Kmeans_cluster
torch.cuda.empty_cache()

model_name = 'wide_resnet50'
# model = models.vgg19(pretrained=True)
model = models.wide_resnet50_2(pretrained=True)
# print(model.__name__)
dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=dataset_mean,
                                 std=dataset_std)

# model_dimension = 299 if args.model == 'InceptionV3' else 256
if model_name =='InceptionV3':
	transform_data = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])	
else:
	transform_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
# trainset = torchvision.datasets.ImageFolder('/mnt/storage0_8/torch_datasets/ILSVRC/train/', transform=transform_data)
# testset = torchvision.datasets.ImageFolder('/mnt/storage0_8/torch_datasets/imagenet_val/', transform=transform_data)
testset = torchvision.datasets.ImageFolder('../../datasets/imagenet_val/', transform=transform_data)

torch.manual_seed(666)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

# label_dict = trainset.class_to_idx
# imagenet_dict = {v:k for k,v in label_dict.items()}
# f = open('labels.txt')
# lines = f.readlines()
# mappingdict = {line.split()[0]:(line.split(',')[0].split('\n')[0].split('\t')[1]).strip(',') for line in lines}


mu = torch.Tensor((dataset_mean)).unsqueeze(-1).unsqueeze(-1).cuda()
std = torch.Tensor((dataset_std)).unsqueeze(-1).unsqueeze(-1).cuda()
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std


for params in model.parameters():
    params.requires_grad = False
model.eval()

# device_ids = [ i for i in range (torch.cuda.device_count())]

# print(device_ids)

# model= nn.DataParallel(model,device_ids=device_ids)

model = model.cuda()
total = 0
correct = 0

km_correct = 0
out_path = './test.png'
for i, (X, y) in enumerate(test_loader):

    X, y = X.cuda(), y.cuda()
    batch_size = X.size(0)

    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
    
    km_X = Kmeans_cluster(X,k=5)
    save_image(km_X, out_path)
    km_outputs = model(km_X)
    _, km_predicted = torch.max(km_outputs.data, 1)
    km_correct += (km_predicted == y).sum()
    # print(f'true label: {y.data}, prediction: {predicted.data}, after km: {km_predicted.data}')
    break

print(f'Kmeans accuracy is {float(km_correct)/total:.4f}! Standard accuracy is {float(correct)/total:.4f}! total: {total}')




