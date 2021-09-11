

import numpy as np
import tensorflow as tf

import math


def box_iou(box1, box2):
    num = box2.shape[-2]
    if len(box1.shape) == 1:
        box1 = np.tile(box1, (num, 1))

    box1_area = np.prod((box1[:, 2:] - box1[:, :2]), axis=-1)
    box2_area = np.prod((box2[..., 2:] - box2[..., :2]), axis=-1)

    left_top = np.maximum(box1[:, :2], box2[..., :2])
    right_bottom = np.minimum(box1[:, 2:], box2[..., 2:])
    inter = np.maximum(right_bottom - left_top, 0.)
    inter_are = np.prod(inter, axis=-1)

    iou = inter_are / (box1_area + box2_area - inter_are)
    return iou


def bbox_iou(boxes1, boxes2, manner='center'):
    if manner == 'center':
        # 变成左上角坐标、右下角坐标
        boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                     boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                     boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    else:
        boxes1_x0y0x1y1 = boxes1
        boxes2_x0y0x1y1 = boxes2
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
            boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
            boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)
    return iou, boxes1_x0y0x1y1, boxes2_x0y0x1y1, union_area


def bbox_giou(boxes1, boxes2):
    iou, boxes1_x0y0x1y1, boxes2_x0y0x1y1, union_area = bbox_iou(boxes1, boxes2)

    c_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    c_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])
    c_inter_section = tf.maximum(c_right_down - c_left_up, 0.0)
    c_inter_area = c_inter_section[..., 0] * c_inter_section[..., 1]

    giou = iou - (c_inter_area - union_area) / (c_inter_area + 1e-9)
    return giou, iou


def bbox_diou(boxes1, boxes2):
    """
    diou = iou - p2/c2
    """
    # 变成左上角坐标、右下角坐标
    iou, boxes1_x0y0x1y1, boxes2_x0y0x1y1, _ = bbox_iou(boxes1, boxes2)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = tf.pow(enclose_wh[..., 0], 2) + tf.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = tf.pow(boxes1[..., 0] - boxes2[..., 0], 2) + tf.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    diou = iou - 1.0 * p2 / (enclose_c2 + 1e-9)
    return diou, iou


def bbox_ciou(boxes1, boxes2):
    """
    ciou = iou - p2/c2 - av = diou - av
    """
    diou, iou = bbox_diou(boxes1, boxes2)

    # 增加av。加上除0保护防止nan。
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * tf.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v + 1e-9)

    ciou = diou - 1.0 * a * v
    return ciou, iou


def matrix_nms():

    def fast_iou(boxes1, boxes2):
        boxes1 = boxes1[:, None, :]
        boxes2 = boxes2[None, :, :]

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_top = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_bottom = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter = tf.maximum(right_bottom - left_top, 0.)
        inter_area = inter[..., 0] * inter[..., 1]

        iou = inter_area / (area1 + area2 - inter_area)
        return iou
    pass
