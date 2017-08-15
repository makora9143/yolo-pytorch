# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DarkNet(nn.Module):
    def __init__(self, pretrain=False):
        super(DarkNet, self).__init__()

        self.pretrain = pretrain

        self.net = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        if self.pretrain:
            self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        output = self.net(x)
        if self.pretrain:
            output = F.avg_pool2d(output, (output.size(2), output.size(3)))
            output = output.squeeze()
            output = F.softmax(self.fc(output))
        return output


class YOLO(nn.Module):
    def __init__(self, model=None, input_size=(480, 640)):
        super(YOLO, self).__init__()
        C = 1

        if model is None:
            model = DarkNet()

        self.darknet = model

        self.yolo = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1)
            )

        self.flatten = Flatten()
        self.fc1 = nn.Linear(math.ceil(input_size[0]/64) * math.ceil(input_size[1] / 64) *1024, 4096)
        self.fc2 = nn.Linear(4096, 7*7*(10 + C))

    def forward(self, x):
        feature = self.darknet(x)
        output = self.yolo(feature)
        output = self.flatten(output)
        output = F.leaky_relu(self.fc1(output), 0.1)
        output = self.fc2(output)

        return output


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

