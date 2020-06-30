# -*- coding: utf-8 -*- 
# @Time : 2020/5/13 13:54 
# @Author : lxd 
# @File : CNN_model.py

# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import os, random, cv2

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.mlp1 = nn.Sequential(
            torch.nn.Linear(in_features=3*12*64, out_features=100),
            nn.ReLU()
        )
        self.mlp2 = torch.nn.Linear(in_features=100, out_features=248)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x

if __name__ == '__main__':
    model = CNN_model()
    # x=torch.randn([1, 3, 30, 100])
    # model(x)
    summary(model, (3,30,100), device='cpu')