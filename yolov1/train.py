# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import cv2

import data
from loss import loss, convert2viz
from model import YOLO


import torchvision.models as models

def train(args):
    print('Dataset of instance(s) and batch size is {}'.format(args.batch_size))
    vgg = models.vgg16(True)
    model = YOLO(vgg.features)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best = 1e+30

    for epoch in range(1, args.epochs+1):
        l = train_epoch(epoch, model, optimizer, args)

        upperleft, bottomright, classes, confs = test_epoch(model, jpg='../data/1.jpg')
        is_best = l < best
        best = min(l, best)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    checkpoint = torch.load('./model_best.pth.tar')
    state_dict = checkpoint['state_dict']

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cpu()

    torch.save(model.state_dict(), 'model_cpu.pth.tar')


def train_epoch(epoch, model, optimizer, args):
    losses = 0.0
    for i, (x, y) in enumerate(data.train_batches(args.batch_size, use_cuda=args.use_cuda), 1):
        optimizer.zero_grad()
        y_pred = model(x)
        l = loss(y_pred, y, use_cuda=args.use_cuda)
        l.backward()
        optimizer.step()
        losses += l.data[0]
    print("Epoch: {}, Ave loss: {}".format(epoch, losses / i))
    return losses / i

def test_epoch(model, threshold, use_cuda=False, jpg=None):
    if jpg is None:
        x = torch.randn(1, 3, 480, 640)
    else:
        img = plt.imread(jpg) / 255.
        x = torch.from_numpy(np.transpose(img, (2, 0, 1)))

    x = Variable(x, requires_grad=False)

    if use_cuda:
        x = x.cuda()

    y = model(x)
    upperleft, bottomright, classes, confs = convert2viz(y)
    result = cv2.imread(jpg)

    for ul, br, cls, cfs in zip(upperleft, bottomright, classes, confs):
        if cfs > threshold:
            result = cv2.rectangle(result, ul, br, (0, 255, 0), 2)
    cv2.imshow(jpg, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth.tar')



