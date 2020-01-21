import torch.nn as nn
import torch
from torch.nn import functional as F

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,64,3,1,1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2,2),
                               nn.Conv2d(64,64,3,1,1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2,2),
                               nn.Conv2d(64,64,3,1,1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2,2),
                               nn.Conv2d(64,64,3,1,1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2,2))
        self.classifier =nn.Linear(64*2*2,10)
        
        
    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x