
from __future__ import division

import numpy as np

from utils.iou import box_iou
from utils.rand import select

import random

"""
 lrtb形式的anchor用于计算与bbox(lrtb)之间的IOU，为positive与negative samples的assign提供判据
 xywh形式的anchor用于计算与bbox(xywh)之间的delta, 作为regression分支的target
"""

def one_pos_ignore_all_neg_encoder(true_boxes,
                                   positive,
                                   search_size,
                                   score_size,
                                   anchors,
                                   high_iou_threshold,
                                   low_iou_threshold):
    """
    标准 anchor-based 检测器的 编码方式
    最大 IOU 单正样本
    小于低 IOU 阈值的全部负样本
    忽略样本两种情况：
    (1) 正负样本以外的 anchors 全部忽略
    (2) 只忽略低 IOU与高 IOU 阈值之间的，相当于大于高 IOU 阈值但不是最大值的 anchors 也会被视为负样本

    :param true_boxes:
    :param positive:
    :param search_size:
    :param score_size:
    :param anchors:
    :param high_iou_threshold:
    :param low_iou_threshold:
    :return:
    """
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    batch = true_boxes.shape[0]
    anchor_num = anchors_xywh.shape[-2]

    cls_label = -1. * np.ones((batch, score_size[0], score_size[1], anchor_num), dtype=np.float32)
    loc_label = np.zeros((batch, score_size[0], score_size[1], anchor_num, 4), dtype=np.float32)

    loc_label[..., 0] = (boxes_xy[..., 0][:, None, None, None] - anchors_xywh[..., 0]) / anchors_xywh[..., 2]
    loc_label[..., 1] = (boxes_xy[..., 1][:, None, None, None] - anchors_xywh[..., 1]) / anchors_xywh[..., 3]
    loc_label[..., 2] = np.log(np.maximum((boxes_wh[..., 0] + 1e-6)[:, None, None, None] / anchors_xywh[..., 2], 1e-6))
    loc_label[..., 3] = np.log(np.maximum((boxes_wh[..., 1] + 1e-6)[:, None, None, None] / anchors_xywh[..., 3], 1e-6))

    for b in range(batch):
        if positive[b]:
            overlap = box_iou(true_boxes[b], all_anchors)

            # ignore = np.where((overlap < high_iou_threshold) & (overlap > low_iou_threshold))
            ignore = np.where(overlap > low_iou_threshold)
            cls_label[b, ignore[0], ignore[1], ignore[2]] = -1.

            max_iou = np.max(overlap)
            if max_iou > high_iou_threshold:
                pos = np.where(overlap == max_iou)
                cls_label[b, pos[0], pos[1], pos[2]] = 1.

    return cls_label, loc_label


def one_pos_ignore_multi_neg_encoder(true_boxes,
                                     positive,
                                     search_size,
                                     score_size,
                                     anchors,
                                     high_iou_threshold,
                                     low_iou_threshold,
                                     easy_neg_num,
                                     mid_neg_num,
                                     hard_neg_num,
                                     k,
                                     radium):
    """
    最大 IOU 单正样本
    小于低 IOU 阈值的 anchors 中抽样得到负样本
    其他 anchors 全部忽略

    :param true_boxes:
    :param positive:
    :param search_size:
    :param score_size:
    :param anchors:
    :param high_iou_threshold:
    :param low_iou_threshold:
    :param easy_neg_num:
    :param mid_neg_num:
    :param hard_neg_num:
    :param k:
    :param radium:
    :return:
    """
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    batch = true_boxes.shape[0]
    anchor_num = anchors_xywh.shape[-2]
    stride = search_size[0] / score_size[0]

    cls_label = -1. * np.ones((batch, score_size[0], score_size[1], anchor_num), dtype=np.float32)
    loc_label = np.zeros((batch, score_size[0], score_size[1], anchor_num, 4), dtype=np.float32)

    loc_label[..., 0] = (boxes_xy[..., 0][:, None, None, None] - anchors_xywh[..., 0]) / anchors_xywh[..., 2]
    loc_label[..., 1] = (boxes_xy[..., 1][:, None, None, None] - anchors_xywh[..., 1]) / anchors_xywh[..., 3]
    loc_label[..., 2] = np.log(np.maximum((boxes_wh[..., 0] + 1e-6)[:, None, None, None] / anchors_xywh[..., 2], 1e-6))
    loc_label[..., 3] = np.log(np.maximum((boxes_wh[..., 1] + 1e-6)[:, None, None, None] / anchors_xywh[..., 3], 1e-6))

    for b in range(batch):
        if positive[b]:
            overlap = box_iou(true_boxes[b], all_anchors)
            neg = np.where(overlap < low_iou_threshold)

            max_iou = np.max(overlap)
            if max_iou > high_iou_threshold:
                pos = np.where(overlap == max_iou)
                cls_label[b, pos[0], pos[1], pos[2]] = 1.

            neg_overlap = overlap[neg]
            # 从IOU < 0.3的前 neg_num * k 的anchor中随机选择难负样本
            hard_neg_index = np.argpartition(neg_overlap, -int(hard_neg_num * k), axis=0)[-int(hard_neg_num * k):]
            hard_neg = tuple(n[hard_neg_index] for n in neg)
            hard_neg, _, hard_neg_index = select(hard_neg, hard_neg_num)

            # 选取分布目标附近可能存在相似物的潜在anchor作为mid负样本
            mid_neg_ = tuple(np.delete(n, hard_neg_index) for n in neg)
            lt = np.maximum(np.ceil((boxes_xy[b, :] - radium * boxes_wh[b, :]) / stride), 0)
            rb = np.minimum(np.floor((boxes_xy[b, :] + radium * boxes_wh[b, :]) / stride), np.array(score_size))
            mid_neg_list = np.stack((mid_neg_[0], mid_neg_[1]), axis=-1)
            mid_neg_index = np.where((mid_neg_list[:, 1] > lt[0]) & (mid_neg_list[:, 0] > lt[1]) &
                                     (mid_neg_list[:, 1] < rb[0]) & (mid_neg_list[:, 0] < rb[1]))
            mid_neg = tuple(n[mid_neg_index] for n in mid_neg_)
            mid_neg, _, mid_neg_index = select(mid_neg, mid_neg_num)

            # 从剩余位置（大部分是边界）随机选择易负样本
            easy_neg = tuple(np.delete(n, mid_neg_index) for n in mid_neg_)
            easy_neg, _, _ = select(easy_neg, easy_neg_num)
            neg_ = tuple(np.concatenate([easy_neg[k], mid_neg[k], hard_neg[k]]) for k in range(3))
            cls_label[b, neg_[0], neg_[1], neg_[2]] = 0.
        else:
            box = true_boxes[b] / stride
            left = max(0, int(np.round(box[0] - 2)))
            up = max(0, int(np.round(box[1] - 2)))
            right = min(score_size[1], int(np.round(box[2] + 2)))
            down = min(score_size[0], int(np.round(box[3] + 2)))

            cls_label[b, up:down, left:right, :] = 0.
            hard_neg, _, _ = select(np.where(cls_label[b, ...] == 0), hard_neg_num)
            cls_label[b, ...] = -1.
            cls_label[b, hard_neg[0], hard_neg[1], hard_neg[2]] = 0.

            easy_neg, _, _ = select(np.where(cls_label[b, ...] == -1), easy_neg_num)
            cls_label[b, easy_neg[0], easy_neg[1], easy_neg[2]] = 0.
    return cls_label, loc_label


def multi_pos_ignore_multi_neg_encoder(true_boxes,
                                       positive,
                                       search_size,
                                       score_size,
                                       anchors,
                                       high_iou_threshold,
                                       pos_num,
                                       easy_pos_num,
                                       hard_pos_num,
                                       low_iou_threshold,
                                       easy_neg_num,
                                       mid_neg_num,
                                       hard_neg_num,
                                       k,
                                       radium):
    """
    Faster RCNN 中 训练 RPN 网络的编码方式
    大于高 IOU 阈值的 anchors 中抽样得到正样本
    小于低 IOU 阈值的 anchors 中抽样得到负样本
    其他 anchors 全部忽略

    :param true_boxes:
    :param positive:
    :param search_size:
    :param score_size:
    :param anchors:
    :param high_iou_threshold:
    :param pos_num:
    :param easy_pos_num:
    :param hard_pos_num:
    :param low_iou_threshold:
    :param easy_neg_num:
    :param mid_neg_num:
    :param hard_neg_num:
    :param k:
    :param radium:
    :return:
    """
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    batch = true_boxes.shape[0]
    anchor_num = anchors_xywh.shape[-2]
    stride = search_size[0] / score_size[0]

    cls_label = -1. * np.ones((batch, score_size[0], score_size[1], anchor_num), dtype=np.float32)
    loc_label = np.zeros((batch, score_size[0], score_size[1], anchor_num, 4), dtype=np.float32)

    loc_label[..., 0] = (boxes_xy[..., 0][:, None, None, None] - anchors_xywh[..., 0]) / anchors_xywh[..., 2]
    loc_label[..., 1] = (boxes_xy[..., 1][:, None, None, None] - anchors_xywh[..., 1]) / anchors_xywh[..., 3]
    loc_label[..., 2] = np.log(np.maximum((boxes_wh[..., 0] + 1e-6)[:, None, None, None] / anchors_xywh[..., 2], 1e-6))
    loc_label[..., 3] = np.log(np.maximum((boxes_wh[..., 1] + 1e-6)[:, None, None, None] / anchors_xywh[..., 3], 1e-6))

    for b in range(batch):
        if positive[b]:
            overlap = box_iou(true_boxes[b], all_anchors)
            neg = np.where(overlap < low_iou_threshold)
            pos = np.where(overlap > high_iou_threshold)
            pos_num_ = pos[0].shape[0]
            if pos_num_ > pos_num:
                pos_overlap = overlap[pos]
                easy_pos_index = np.argpartition(pos_overlap, -easy_pos_num, axis=0)[-easy_pos_num:]
                easy_pos = tuple(p[easy_pos_index] for p in pos)

                hard_pos_index = np.argpartition(pos_overlap, hard_pos_num, axis=0)[hard_pos_num:]
                hard_pos = tuple(p[hard_pos_index] for p in pos)
                pos_ = tuple(np.concatenate([easy_pos[c], hard_pos[c]]) for c in range(3))
            else:
                if pos_num_ == 0:
                    max_iou = np.max(overlap)
                    pos_ = np.where(overlap == max_iou)
                else:
                    pos_ = pos

            if hard_neg_num > 0:
                neg_overlap = overlap[neg]
                # 从IOU < 0.3的前 neg_num * k 的anchor中随机选择难负样本
                hard_neg_index = np.argpartition(neg_overlap, -int(hard_neg_num * k), axis=0)[
                                 -int(hard_neg_num * k):]
                hard_neg = tuple(n[hard_neg_index] for n in neg)
                hard_neg, _, hard_neg_index = select(hard_neg, hard_neg_num)

                # 选取分布目标附近可能存在相似物的潜在anchor作为mid负样本
                mid_neg_ = tuple(np.delete(n, hard_neg_index) for n in neg)
            else:
                mid_neg_ = neg

            if mid_neg_num > 0:
                lt = np.maximum(np.ceil((boxes_xy[b, :] - radium * boxes_wh[b, :]) / stride), 0)
                rb = np.minimum(np.floor((boxes_xy[b, :] + radium * boxes_wh[b, :]) / stride), np.array(score_size))
                mid_neg_list = np.stack((mid_neg_[0], mid_neg_[1]), axis=-1)
                mid_neg_index = np.where((mid_neg_list[:, 1] > lt[0]) & (mid_neg_list[:, 0] > lt[1]) &
                                         (mid_neg_list[:, 1] < rb[0]) & (mid_neg_list[:, 0] < rb[1]))

                mid_neg = tuple(n[mid_neg_index] for n in mid_neg_)
                mid_neg, _, mid_neg_index = select(mid_neg, mid_neg_num)

                # 从剩余位置（大部分是边界）随机选择易负样本
                easy_neg = tuple(np.delete(n, mid_neg_index) for n in mid_neg_)
            else:
                if hard_neg_num > 0:
                    easy_neg = mid_neg_
                else:
                    easy_neg = neg

            easy_neg, _, _ = select(easy_neg, easy_neg_num)

            if hard_neg_num > 0 and mid_neg_num > 0:
                neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c], hard_neg[c]]) for c in range(3))
            if hard_neg_num > 0 and mid_neg_num == 0:
                neg_ = tuple(np.concatenate([easy_neg[c], hard_neg[c]]) for c in range(3))
            if hard_neg_num == 0 and mid_neg_num > 0:
                neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c]]) for c in range(3))
            if hard_neg_num == 0 and mid_neg_num == 0:
                neg_ = easy_neg

            cls_label[b, neg_[0], neg_[1], neg_[2]] = 0.
            cls_label[b, pos_[0], pos_[1], pos_[2]] = 1.
        else:
            box = true_boxes[b] / stride
            left = max(0, int(np.round(box[0] - 2)))
            up = max(0, int(np.round(box[1] - 2)))
            right = min(score_size[1], int(np.round(box[2] + 2)))
            down = min(score_size[0], int(np.round(box[3] + 2)))

            cls_label[b, up:down, left:right, :] = 0.
            hard_neg, _, _ = select(np.where(cls_label[b, ...] == 0.), hard_neg_num)
            cls_label[b, ...] = -1.
            cls_label[b, hard_neg[0], hard_neg[1], hard_neg[2]] = 0.

            easy_neg, _, _ = select(np.where(cls_label[b, ...] == -1.), easy_neg_num)
            cls_label[b, easy_neg[0], easy_neg[1], easy_neg[2]] = 0.
    return cls_label, loc_label


def anchor_based_mix_encoder(true_boxes,
                             mix_boxes_,
                             positive,
                             search_size,
                             score_size,
                             anchors,
                             high_iou_threshold,
                             pos_num,
                             easy_pos_num,
                             hard_pos_num,
                             low_iou_threshold,
                             easy_neg_num,
                             mid_neg_num,
                             hard_neg_num,
                             k,
                             radium):
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    batch = true_boxes.shape[0]
    anchor_num = anchors_xywh.shape[-2]
    stride = search_size[0] / score_size[0]
    # 空的classification label为与输出形状相同的全-1矩阵，完成assign后，正样本处值为1，负样本处值为0，仍为-1的位置在损失计算中忽略
    cls_label = -1. * np.ones((batch, score_size[0], score_size[1], anchor_num), dtype=np.float32)
    loc_label = np.zeros((batch, score_size[0], score_size[1], anchor_num, 4), dtype=np.float32)

    # encode: 一次性编码出所有anchor相对于bbox的偏移量，作为回归分支的target
    loc_label[..., 0] = (boxes_xy[..., 0][:, None, None, None] - anchors_xywh[..., 0]) / anchors_xywh[..., 2]
    loc_label[..., 1] = (boxes_xy[..., 1][:, None, None, None] - anchors_xywh[..., 1]) / anchors_xywh[..., 3]
    loc_label[..., 2] = np.log(np.maximum((boxes_wh[..., 0] + 1e-6)[:, None, None, None] / anchors_xywh[..., 2], 1e-6))
    loc_label[..., 3] = np.log(np.maximum((boxes_wh[..., 1] + 1e-6)[:, None, None, None] / anchors_xywh[..., 3], 1e-6))

    for b in range(batch):
        # 计算出所有anchor与bbox之间的IOU值，初步筛选出高IOU、低IOU与IOU最高的anchors
        overlap = box_iou(true_boxes[b], all_anchors)
        neg = np.where(overlap < low_iou_threshold)
        pos = np.where(overlap > high_iou_threshold)
        max_iou = np.max(overlap)
        max_pos = np.where(overlap == max_iou)

        # 无论是正图像对还是负图像对，其他物体的高IOU anchors必定设为负样本。
        # 由于会存在完全覆盖目标或与目标具有极高重合度的物体，有可能会覆盖掉目标的正anchors，因此需要用最高IOU的anchor作为保底。
        # 由于干扰物体的尺寸差异很大，相当一部分干扰box没有适配的anchor，同样采取保底措施，采用最高IOU的anchor，但是对IOU大小有一定要求
        mix_boxes = mix_boxes_[b]
        mix_neg_num = 0
        if mix_boxes is not None:
            mix_num = mix_boxes.shape[0]
            for i in range(mix_num):
                max_num = random.randint(6, 10)
                mix_box = mix_boxes[i, :]
                mix_overlap = box_iou(mix_box, all_anchors)
                mix = np.where(mix_overlap > 0.45)
                mix_neg_num_ = mix[0].shape[0]

                if mix_neg_num_ > max_num:
                    pos_overlap_ = mix_overlap[mix]
                    pos_index = np.argpartition(pos_overlap_, -max_num, axis=0)[-max_num:]
                    mix = tuple(m[pos_index] for m in mix)
                else:
                    if mix_neg_num_ == 0:
                        max_iou = np.max(mix_overlap)
                        if max_iou > 0.25:
                            mix = np.where(mix_overlap == max_iou)

                cls_label[b, mix[0], mix[1], mix[2]] = 0.

                mix_neg_num += mix[0].shape[0]

        # 高IOU的anchor的数量超过设定数量上限时，进行筛选选取；
        # 未超过数量上限时，全部留下；
        # 没有符合条件的anchor时，选取最大IOU的anchor
        pos_num_ = pos[0].shape[0]
        if pos_num_ > pos_num:
            pos_overlap = overlap[pos]
            easy_pos_index = np.argpartition(pos_overlap, -easy_pos_num, axis=0)[-easy_pos_num:]
            easy_pos = tuple(p[easy_pos_index] for p in pos)

            hard_pos_index = np.argpartition(pos_overlap, hard_pos_num, axis=0)[:hard_pos_num]
            hard_pos = tuple(p[hard_pos_index] for p in pos)
            pos_ = tuple(np.concatenate([easy_pos[c], hard_pos[c]]) for c in range(3))
        else:
            if pos_num_ > 0:
                pos_ = pos
            else:
                pos_ = max_pos

        # 正图像对中，负样本anchor从低IOU的anchor中随机选取，按照与目标的距离分为难、中、易三个档次
        # 负图像对中，与bbox高IOU的anchors必定为负样本，另外也从低IOU的anchor中随机选取负样本，同样按照与目标的距离分为难、中、易三个档次
        if positive[b]:
            if hard_neg_num > 0:
                neg_overlap = overlap[neg]
                # 从IOU < 0.3的前 neg_num * k 的anchor中随机选择难负样本
                hard_neg_index = np.argpartition(neg_overlap, -int(hard_neg_num * k), axis=0)[-int(hard_neg_num * k):]
                hard_neg = tuple(n[hard_neg_index] for n in neg)
                hard_neg, _, hard_neg_index = select(hard_neg, hard_neg_num)

                # 选取分布目标附近可能存在相似物的潜在anchor作为mid负样本
                mid_neg_ = tuple(np.delete(n, hard_neg_index) for n in neg)
            else:
                mid_neg_ = neg

            if mid_neg_num > 0:
                lt = np.maximum(np.ceil((boxes_xy[b, :] - radium * boxes_wh[b, :]) / stride), 0)
                rb = np.minimum(np.floor((boxes_xy[b, :] + radium * boxes_wh[b, :]) / stride), np.array(score_size))
                mid_neg_list = np.stack((mid_neg_[0], mid_neg_[1]), axis=-1)
                mid_neg_index = np.where((mid_neg_list[:, 1] > lt[0]) & (mid_neg_list[:, 0] > lt[1]) &
                                         (mid_neg_list[:, 1] < rb[0]) & (mid_neg_list[:, 0] < rb[1]))

                mid_neg = tuple(n[mid_neg_index] for n in mid_neg_)
                mid_neg, _, mid_neg_index = select(mid_neg, mid_neg_num)

                # 从剩余位置（大部分是边界）随机选择易负样本
                easy_neg = tuple(np.delete(n, mid_neg_index) for n in mid_neg_)
            else:
                if hard_neg_num > 0:
                    easy_neg = mid_neg_
                else:
                    easy_neg = neg

            easy_neg, _, _ = select(easy_neg, easy_neg_num)

            if hard_neg_num > 0 and mid_neg_num > 0:
                neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c], hard_neg[c]]) for c in range(3))
            if hard_neg_num > 0 and mid_neg_num == 0:
                neg_ = tuple(np.concatenate([easy_neg[c], hard_neg[c]]) for c in range(3))
            if hard_neg_num == 0 and mid_neg_num > 0:
                neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c]]) for c in range(3))
            if hard_neg_num == 0 and mid_neg_num == 0:
                neg_ = easy_neg

            neg_num = neg_[0].shape[0]
            neg_num_ = neg_num - mix_neg_num
            if neg_num_ < 8:
                neg_num_ = random.randint(6, 10)
            neg_, _, _ = select(neg_, neg_num_)

            cls_label[b, neg_[0], neg_[1], neg_[2]] = 0.
            cls_label[b, pos_[0], pos_[1], pos_[2]] = 1.

        else:
            neg_num_ = hard_neg_num + mid_neg_num + easy_neg_num - pos_[0].shape[0] - mix_neg_num
            if neg_num_ < 8:
                neg_num_ = random.randint(6, 10)

            cls_label[b, pos_[0], pos_[1], pos_[2]] = 0.

            easy_neg, _, _ = select(np.where(cls_label[b, ...] == -1.), neg_num_)
            cls_label[b, easy_neg[0], easy_neg[1], easy_neg[2]] = 0.

    return cls_label, loc_label
