import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def Le_Net():
    model = nn.Sequential(
        nn.Conv2d(1, 25, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(25, 50, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(50*4*4,100),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(100, 10)
    )
    return model

def classifier_A():
    model = nn.Sequential(
        nn.Conv2d(1, 64, 5),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5),
        nn.ReLU(),
        nn.Dropout2d(p=0.25),
        Flatten(),
        nn.Linear(64*20*20,128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 10)
    )
    return model

def classifier_B():
    model = nn.Sequential(
        nn.Dropout2d(p=0.2),
        nn.Conv2d(1,64,8),
        nn.ReLU(),
        nn.Conv2d(64,128,6),
        nn.ReLU(),
        nn.Conv2d(128,128,5),
        nn.ReLU(),
        nn.Dropout2d(p=0.5),
        Flatten(),
        nn.Linear(128*12*12, 10)
    )
    return model

def classifier_C():
    model = nn.Sequential(
        nn.Conv2d(1, 128, 3),
        nn.Tanh(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 64, 3),
        nn.Tanh(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(64*5*5,128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model