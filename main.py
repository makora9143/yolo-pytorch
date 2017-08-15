# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import argparse
import train


def main():
    parser = argparse.ArgumentParser(description='PyTorch YOLO')

    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='use cuda or not')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()
    train.train(args)




if __name__ == '__main__':
    main()
