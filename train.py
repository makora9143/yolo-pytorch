# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data
from loss import loss
from model import YOLO

def train(args):
    epochs = 3
    batch_size = 2

    model = YOLO()

    for epoch in range(epochs):
        for x, y in data.train_batches(batch_size):
            y_pred = model(x)
            l = loss(y_pred, y)
            print(l.data[0])
            l.backward()
