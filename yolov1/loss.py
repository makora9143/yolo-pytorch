# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt


def loss(y_pred, y_true, S=7, B=2, C=1, use_cuda=False):
    ''' Calculate the loss of YOLO model.
    args:
        y_pred: (Batch, 7 * 7 * 30)
        y_true: dict object that contains:
            class_probs,
            confs,
            coord,
            proid,
            areas,
            upleft,
            bottomright

    '''

    SS = S * S
    scale_class_prob = 1
    scale_object_conf = 1
    scale_noobject_conf = 0.5
    scale_coordinate = 5
    batch_size = y_pred.size(0)

    # ground truth
    _coord = y_true['coord']
    _coord = _coord.view(-1, SS, B, 4)
    _upleft = y_true['upleft']
    _bottomright = y_true['bottomright']
    _areas = y_true['areas']
    _confs = y_true['confs']
    _proid = y_true['proid']
    _probs = y_true['class_probs']

    # Extract the coordinate prediction from y_pred
    coords = y_pred[:, SS * (C + B):].contiguous().view(-1, SS, B, 4)
    wh = torch.pow(coords[:, :, :, 2:4], 2)
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2].contiguous()
    floor = centers - (wh * 0.5)
    ceil = centers + (wh * 0.5)

    # Calculate the intersection areas
    intersect_upleft = torch.max(floor, _upleft)
    intersect_bottomright = torch.min(ceil, _bottomright)
    intersect_wh = intersect_bottomright - intersect_upleft
    zeros = Variable(torch.zeros(batch_size, SS, B, 2)).cuda() if use_cuda else Variable(torch.zeros(batch_size, 49, B, 2))
    intersect_wh = torch.max(intersect_wh, zeros)
    intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]

    # Calculate the best IOU, set 0.0 confidence for worse boxes
    iou = intersect / (_areas + area_pred - intersect)
    best_box = torch.eq(iou, torch.max(iou, 2)[0].unsqueeze(2))
    confs = best_box.float() * _confs

    # Take care of the weight terms
    conid = scale_noobject_conf * (1. - confs) + scale_object_conf * confs
    weight_coo = torch.cat(4 * [confs.unsqueeze(-1)], 3)
    cooid = scale_coordinate * weight_coo
    proid = scale_class_prob * _proid

    # Flatten 'em all
    probs = flatten(_probs)
    proid = flatten(proid)
    confs = flatten(confs)
    conid = flatten(conid)
    coord = flatten(_coord)
    cooid = flatten(cooid)

    true = torch.cat([probs, confs, coord], 1)
    wght = torch.cat([proid, conid, cooid], 1)

    loss = torch.pow(y_pred - true, 2)
    loss = loss * wght
    loss = torch.sum(loss, 1)
    return .5 * torch.mean(loss)

def flatten(x):
    return x.view(x.size(0), -1)

def convert2viz(y_pred, C=1, B=2, S=7):
    SS = S * S
    coords = y_pred[:, SS * (C + B):].contiguous().view(-1, SS, B, 4)
    wh = torch.pow(coords[:, :, :, 2:4], 2)
    xy = coords[:, :, :, 0:2]
    upperleft = xy - wh / 2
    bottomright = xy + wh / 2
    classes = y_pred[:, :SS * C].contiguous().view(-1, SS, C)
    classes = classes.unsqueeze(2).repeat(1, 1, B, 1)
    confs = y_pred[:, SS * C: SS * (C+B)].contiguous().view(-1, SS, B)

    return upperleft.view(-1, SS*B, 2), bottomright.view(-1, SS*B, 2), classes.view(-1, SS*B, C), confs.view(-1, SS*B)

