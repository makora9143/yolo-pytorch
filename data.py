# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import json
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def create_label(chunk):
    '''
    input: chunk
    chunk is list object.
        -> [img_path, [w, h, [[label, xn, yn, xx, yx], [label, xn, yn, xx, yx], ..., [label, xn, yn, xx, yx]]]

            img_path: "path_to_img/sample.jpg",
            w: image width,
            h: image height,
            label: object class,
            xn, yn: topleft coordinates,
            xx, yx: bottomright coordinates

    For example:
    if the data is:
        {
            {"img": "path_to_img/sample.jpg", w: 640, h: 480},
            {"label": "person", "topleft": {"x": 189, "y": 96}, "bottomright": {"x": 271, "y": 380}},
            {"label": "dog", "topleft": {"x": 69, "y": 258}, "bottomright": {"x": 209, "y": 354}},
            {"label": "horse", "topleft": {"x": 397, "y": 127}, "bottomright": {"x": 605, "y": 352}},
        }
    then:
        [
            "path_to_img/sample.jpg",
            [
                640,
                480,
                [
                    ['person', 189, 96, 271, 380],
                    ['dog', 69, 258, 209, 354],
                    ['horse', 397, 127, 605, 352]
                ],
            ]
        ]

    output: torch.FloatTensor
        (Batch size, 7 * 7 * 30)

    '''
    S, B = 7, 2
    C = 1
    labels = {
            "gomi": 0,
            }

    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    img = plt.imread(jpg) / 255.
    img = np.transpose(img, (2, 0, 1))

    cell_x = 1. * w / S # width per cell
    cell_y = 1. * h / S # height per cell

    for obj in allobj:
        # center_x = 0.5 * (obj[1] + obj[3]) # (xmin + xmax) / 2
        # center_y = 0.5 * (obj[2] + obj[4]) # (ymin + ymax) / 2
        center_x = obj[1] # (xmin + xmax) / 2
        center_y = obj[2] # (ymin + ymax) / 2

        cx = center_x / cell_x # rescale the center x to cell size
        cy = center_y / cell_y # rescale the center y to cell size
        if cx >= S or cy >= S: return None, None

        # obj[3] = float(obj[3] - obj[1]) / w # calculate and normalize width
        # obj[4] = float(obj[4] - obj[2]) / h # calculate and normalize height
        obj[3] = obj[3] / w # calculate and normalize width
        obj[4] = obj[4] / h # calculate and normalize height
        obj[3] = np.sqrt(obj[3]) # sqrt w
        obj[4] = np.sqrt(obj[4]) # sqrt h

        obj[1] = cx - np.floor(cx) # center x in each cell
        obj[2] = cy - np.floor(cy) # center x in each cell

        obj += [int(np.floor(cy) * S + np.floor(cx))] # indexing cell[0, 49)

    # each object: length: 6,
    # [label, center_x_in_cell, center_y_in_cell, w_in_image, h_in_image, cell_idx]

    class_probs = np.zeros([S*S, C]) # for one_hot vector per each cell
    confs = np.zeros([S*S, B]) # for 2 bounding box per each cell
    coord = np.zeros([S*S, B, 4]) # for 4 coordinates per bounding box per cell
    proid = np.zeros([S*S, C]) # for class_probs weight \mathbb{1}^{obj}
    prear = np.zeros([S*S, 4]) # for bounding box coordinates

    for obj in allobj:
        class_probs[obj[5], :] = [0.] * C # no need?
        class_probs[obj[5], labels[obj[0]]] = 1.

        # for object confidence? -> the cell which contains object is 1 nor 0
        confs[obj[5], :] = [1.] * B 

        # assign [center_x_in_cell, center_y_in_cell, w_in_image, h_in_image]
        coord[obj[5], :, :] = [obj[1:5]] * B 

        # for 1_{i}^{obj} in paper eq.(3)
        proid[obj[5], :] = [1] * C

        # transform width and height to the scale of coordinates
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * 0.5 * S # x_left
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * 0.5 * S # y_top
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * 0.5 * S # x_right
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * 0.5 * S # y_bottom

    # for calculate upleft, bottomright and areas for 2 bounding box(not for 1 bounding box)
    upleft = np.expand_dims(prear[:, 0:2], 1)
    bottomright = np.expand_dims(prear[:, 2:4], 1)
    wh = bottomright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    bottomright = np.concatenate([bottomright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    y_true = {
            'class_probs': class_probs,
            'confs': confs,
            'coord': coord,
            'proid': proid,
            'areas': areas,
            'upleft': upleft,
            'bottomright': bottomright
            }
    return img, y_true


def load_data(i):
    img = './data/{}.jpg'.format(i)

    with open('./data/annotations/{}.json'.format(i)) as f:
        json_dict = json.load(f)

    wh = json_dict['image_w_h']
    allobj = [[obj['label']] + obj['x_y_w_h']
               for obj in json_dict['objects']]
    wh.append(allobj)

    return [img, wh]


def get_datas(idx, use_cuda=False):
    x_batch = list()
    feed_batch = dict()
    for i in idx:
        chunk = load_data(i)
        img, new_feed = create_label(chunk)

        if img is None:
            continue
        x_batch += [np.expand_dims(img, 0)]

        for key in new_feed:
            new = new_feed[key]
            old_feed = feed_batch.get(key,
                    np.zeros((0,) + new.shape))

            feed_batch[key] = np.concatenate([
                old_feed, [new]])

    if use_cuda:
        x_batch = Variable(torch.from_numpy(np.concatenate(x_batch, 0)).float()).cuda()
        feed_batch = {key: Variable(torch.from_numpy(feed_batch[key]).float()).cuda()
                      for key in feed_batch}

    else:
        x_batch = torch.from_numpy(np.concatenate(x_batch, 0)).float()
        feed_batch = {key: Variable(torch.from_numpy(feed_batch[key]).float())
                      for key in feed_batch}

    return x_batch, feed_batch


def train_batches(batch_size=1, train_size=6, use_cuda=False):
    print('Dataset of {} instance(s) and batch size is {}'.format(train_size, batch_size))
    shuffle_idx = np.random.permutation(list(range(1, train_size + 1)))
    for i in range(train_size // batch_size):
        yield get_datas(shuffle_idx[i*batch_size: (i+1)*batch_size], use_cuda)

