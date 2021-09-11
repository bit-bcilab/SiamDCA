

import numpy as np

from utils.rand import select
from utils.bbox import corner2center

import random


"""
Assign positive and negative samples based on ellipse
Regression target is the distance between the point on grid and groundtruth bounding box (LTRB format) 

Refer to SiamBAN ———— Siamese Box Adaptive Network for Visual Tracking
Paper: https://arxiv.org/pdf/2003.06761.pdf
Code: https://github.com/hqucv/siamban
"""


def pos_assign(bbox, grid, radium=4., max_num=16):
    tcx, tcy, tw, th = bbox

    # 中心区域为正
    pos_ = np.where((np.square(tcx - grid[..., 0]) / (np.square(tw / radium) + 1e-5) +
                     np.square(tcy - grid[..., 1]) / (np.square(th / radium) + 1e-5)) <= 1.)
    pos_num = pos_[0].shape[0]

    # 数量超过上限时，随机挑选
    if pos_num > max_num:
        pos, _, _ = select(pos_, max_num)
        pos_num = max_num
    else:
        if pos_num > 0:
            pos = pos_
        # 没有满足条件的点时，选择离中心最近的点
        else:
            center_x = grid[0, :, 0]
            center_y = grid[:, 0, 1]
            x_pos = np.argmin(np.abs(center_x - tcx))
            y_pos = np.argmin(np.abs(center_y - tcy))
            pos = (y_pos, x_pos)
            pos_num = 1
    return pos, pos_num


def neg_assign(bbox, grid, radium=2., max_num=48):
    tcx, tcy, tw, th = bbox

    neg_ = np.where((np.square(tcx - grid[..., 0]) / (np.square(tw / radium) + 1e-5) +
                     np.square(tcy - grid[..., 1]) / (np.square(th / radium) + 1e-5)) > 1.)
    neg_num = neg_[0].shape[0]

    # 数量超过上限时，随机挑选
    if neg_num > max_num:
        neg, _, _ = select(neg_, max_num)
        neg_num = max_num
    else:
        neg = neg_
    return neg, neg_num


def ltrb_encoder(true_boxes,
                 positive,
                 search_size,
                 score_size,
                 grid,
                 pos_num,
                 neg_num,
                 pos_radium=4.,
                 neg_radium=2.):
    batch = true_boxes.shape[0]
    grid_ = grid[None, ...]
    true_boxes_ = true_boxes[:, None, None, :]
    boxes_xywh = corner2center(true_boxes)

    cls_label = -1. * np.ones((batch, score_size[0], score_size[1]), dtype=np.float32)
    loc_target = np.zeros((batch, score_size[0], score_size[1], 4), dtype=np.float32)

    loc_target[..., :2] = grid_ - true_boxes_[..., :2]
    loc_target[..., 2:] = true_boxes_[..., 2:] - grid_

    for b in range(batch):
        bbox = boxes_xywh[b]

        if positive[b]:
            neg, neg_num_ = neg_assign(bbox, grid, radium=neg_radium, max_num=random.randint(neg_num - 5, neg_num + 5))
            pos, pos_num_ = pos_assign(bbox, grid, radium=pos_radium, max_num=random.randint(pos_num - 2, pos_num + 2))

            cls_label[b, neg[0], neg[1]] = 0.
            cls_label[b, pos[0], pos[1]] = 1.
        else:
            # 负样本对时，在原目标位置选取大量负样本点，迫使网络进行辨别，而不是记忆
            max_num = int(1.5 * pos_num)
            pos, pos_num_ = pos_assign(bbox, grid, radium=neg_radium, max_num=random.randint(max_num - 4, max_num + 4))
            neg, neg_num_ = neg_assign(bbox, grid, radium=neg_radium, max_num=int(pos_num_ // 2))

            cls_label[b, neg[0], neg[1]] = 0.
            cls_label[b, pos[0], pos[1]] = 0.

    return cls_label, loc_target


def ltrb_mix_encoder(true_boxes,
                     mix_boxes_,
                     positive,
                     search_size,
                     score_size,
                     grid,
                     pos_num,
                     neg_num,
                     pos_radium=4.,
                     neg_radium=2.):
    batch = true_boxes.shape[0]
    grid_ = grid[None, ...]
    true_boxes_ = true_boxes[:, None, None, :]
    boxes_xywh = corner2center(true_boxes)

    cls_label = -1. * np.ones((batch, score_size[0], score_size[1]), dtype=np.float32)
    loc_target = np.zeros((batch, score_size[0], score_size[1], 4), dtype=np.float32)

    loc_target[..., :2] = grid_ - true_boxes_[..., :2]
    loc_target[..., 2:] = true_boxes_[..., 2:] - grid_

    for b in range(batch):
        mix_radium = (pos_radium + neg_radium) / 2.
        mix_boxes = mix_boxes_[b]
        mix_neg_num = 0
        if mix_boxes is not None:
            mix_num = mix_boxes.shape[0]
            mix_xywh = corner2center(mix_boxes)
            for i in range(mix_num):
                mix, mix_neg_num_ = pos_assign(mix_xywh[i], grid, radium=mix_radium, max_num=random.randint(10, 14))

                cls_label[b, mix[0], mix[1]] = 0.
                mix_neg_num += mix_neg_num_

        bbox = boxes_xywh[b]

        if positive[b]:
            if mix_neg_num > 0:
                neg_num_ = random.randint(10, 12)
            else:
                neg_num_ = neg_num
            neg, neg_num_ = neg_assign(bbox, grid, radium=neg_radium, max_num=neg_num_)
            pos, pos_num_ = pos_assign(bbox, grid, radium=pos_radium, max_num=random.randint(pos_num - 2, pos_num + 2))

            cls_label[b, neg[0], neg[1]] = 0.
            cls_label[b, pos[0], pos[1]] = 1.
        else:
            max_num = int(1.5 * pos_num)
            pos, pos_num_ = pos_assign(bbox, grid,  radium=neg_radium, max_num=random.randint(max_num - 4, max_num + 4))
            neg, neg_num_ = neg_assign(bbox, grid, radium=neg_radium, max_num=random.randint(6, 10))

            cls_label[b, neg[0], neg[1]] = 0.
            cls_label[b, pos[0], pos[1]] = 0.

    return cls_label, loc_target


def ltrb_self_encoder(true_boxes,
                      positive,
                      search_size,
                      score_size,
                      grid,
                      pos_radium=4.,
                      neg_radium=1.6):
    batch = true_boxes.shape[0]
    grid_ = grid[None, ...]
    true_boxes_ = true_boxes[:, None, None, :]
    boxes_xywh = corner2center(true_boxes)

    mask = np.ones((batch, score_size[0], score_size[1]), dtype=np.float32)
    cls_label = -1. * np.ones((batch, score_size[0], score_size[1]), dtype=np.float32)
    loc_target = np.zeros((batch, score_size[0], score_size[1], 4), dtype=np.float32)

    loc_target[..., :2] = grid_ - true_boxes_[..., :2]
    loc_target[..., 2:] = true_boxes_[..., 2:] - grid_

    for b in range(batch):
        if positive[b]:
            bbox = boxes_xywh[b]

            pos_mask, _ = pos_assign(bbox, grid, radium=neg_radium, max_num=10000)  # 1.6
            mask[b, pos_mask[0], pos_mask[1]] = 0.

            pos, _ = pos_assign(bbox, grid, radium=pos_radium, max_num=10000)
            cls_label[b, pos[0], pos[1]] = 1.

    return mask, cls_label, loc_target
