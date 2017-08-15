# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import data
from loss import loss
from model import YOLO


import torchvision.models as models

def train(args):
    vgg = models.vgg16(True)
    model = YOLO(vgg.features).cuda() if args.use_cuda else YOLO()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for x, y in data.train_batches(args.batch_size, use_cuda=args.use_cuda):
            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y, use_cuda=args.use_cuda)
            l.backward()
            optimizer.step()
            print(l.data[0])

def pretrain():
    pass
