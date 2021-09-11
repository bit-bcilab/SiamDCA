

import numpy as np
import tensorflow as tf
from keras.layers import Layer

from utils.iou import bbox_iou

import random


class LabelUpdate(Layer):
    def __init__(self, num_pos, num_easy_neg, num_hard_neg, threshold=0.3, **kwargs):
        super(LabelUpdate, self).__init__(**kwargs)
        self.num_pos = num_pos
        self.num_easy_neg = num_easy_neg
        self.num_hard_neg = num_hard_neg
        self.threshold = threshold

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super(LabelUpdate, self).get_config()
        base_config.update({'num_pos': self.num_pos, 'threshold': self.threshold,
                            'num_easy_neg': self.num_easy_neg, 'num_hard_neg': self.num_hard_neg})
        return base_config

    def call(self, inputs, **kwargs):

        # 与目标的IOU高于0.25的所有anchor对应的值为0，其余位置都为1
        # 用于滤去得分图中正类的得分
        # shape = (b, 32, 32)
        mask = inputs[0]
        # shape = (b, 32, 32)
        pos_label = inputs[1]
        # shape = (b, 32, 32, 2)
        pred = inputs[2]

        pred = tf.stop_gradient(pred)

        # shape = (b, 32, 32)
        score = tf.nn.softmax(pred, axis=-1)[..., 1]

        # 提取出所有得分高于th的点
        score = tf.cast(tf.greater_equal(score, 0.3), dtype=tf.float32)

        # mask掉高得分中的正样本点得分
        # 剩余的认为都是异常高得分，要作为负样本
        neg_label = mask * score

        # 得分图的得分阈值取高点，mask的IOU阈值取低点，避免误伤，即将正样本点视为负样本

        # shape = (b, 32, 32)
        empty_label = tf.ones_like(mask, dtype=tf.float32)

        # 在所有异常得分点中进行抽样，得到难负样本，目的是提升对相似物体的辨别能力
        hard_neg_label = tf.py_func(self.random_choice, [neg_label, random.randint(32, 36)], [tf.float32])[0]
        hard_neg_label = tf.reshape(hard_neg_label, shape=tf.shape(mask))

        # 在所有低IOU anchor中随机抽取易负样本，保证对背景的辨别力
        easy_neg_label = tf.py_func(self.random_choice, [mask, random.randint(12, 18)], [tf.float32])[0]
        easy_neg_label = tf.reshape(easy_neg_label, shape=tf.shape(mask))

        # 有可能出现重复取样，因此需要进行以下整理
        neg_label = hard_neg_label + easy_neg_label
        neg_label = tf.cast(tf.greater_equal(neg_label, 1.), dtype=tf.float32)

        # 高IOU anchor中同样进行抽样，得到正样本
        pos_label = tf.py_func(self.random_choice, [pos_label, random.randint(16, 20)], [tf.float32])[0]
        pos_label = tf.reshape(pos_label, shape=tf.shape(mask))

        label = -1. * empty_label + 1. * neg_label + 2. * pos_label
        label = tf.stop_gradient(label)
        return label

    @staticmethod
    def random_choice(label, max_num):
        # shape = (b, 32, 32)
        label_ = np.zeros_like(label)
        batch = label.shape[0]
        for i in range(batch):
            index = np.where(label[i, ...] == 1.)
            num = index[0].shape[0]
            if num > max_num:
                slt = np.arange(num)
                random.shuffle(slt)
                slt = slt[:max_num]
                index = tuple(p[slt] for p in index)
            if num > 0:
                label_[i, index[0], index[1]] = 1.
        return label_


class LabelDelta(Layer):
    def __init__(self, **kwargs):
        super(LabelDelta, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        roi_xywh = inputs[0]
        boxes_xywh = inputs[1]

        delta_xy = (boxes_xywh[..., :2] - roi_xywh[..., :2]) / tf.maximum(roi_xywh[..., 2:], 1e-4)
        delta_wh = tf.log(tf.maximum((boxes_xywh[..., 2:] + 1e-6) / tf.maximum(roi_xywh[..., 2:], 1e-4), 1e-6))
        delta = tf.concat([delta_xy, delta_wh], axis=-1)
        return delta


class LabelCls(Layer):
    def __init__(self, num_cascade, num_roi, num_pos, num_neg, iou_thresh,
                 pos_overlap_thresh, neg_overlap_thresh, max_pos=True, **kwargs):
        super(LabelCls, self).__init__(**kwargs)
        self.num_cascade = num_cascade
        self.num_roi = num_roi
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.iou_thresh = iou_thresh
        self.pos_overlap_thresh = pos_overlap_thresh
        self.neg_overlap_thresh = neg_overlap_thresh
        self.max_pos = max_pos

    def get_config(self):
        base_config = super(LabelCls, self).get_config()
        base_config.update({'num_pos': self.num_pos, 'num_neg': self.num_neg, 'iou_thresh': self.iou_thresh, 'max_pos': self.max_pos,
                            'pos_overlap_thresh': self.pos_overlap_thresh, 'neg_overlap_thresh': self.neg_overlap_thresh})
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        self.head_index = 0
        if self.num_cascade == 1:
            label = self.cls_label(inputs[0], inputs[1], inputs[2])
        else:
            label = []
            for i in range(self.num_cascade):
                start = i * self.num_roi
                end = (i + 1) * self.num_roi
                overlap = inputs[0][:, start:end, ...]
                iou = inputs[1][:, start:end, ...]
                neg_mask = inputs[2][:, start:end, ...]
                label.append(self.cls_label(overlap, iou, neg_mask))
                self.head_index += 1
            label = tf.concat(label, axis=1)
        return label

    def cls_label(self, overlap, iou, neg_mask):
        # 在 overlap 大于阈值 的 ROI 中筛选正样本
        pos_mask = tf.cast(tf.greater_equal(overlap, self.pos_overlap_thresh[self.head_index]), tf.float32)

        # shape = (b, 1)
        max_iou = tf.reduce_max(iou, axis=1, keepdims=True)
        # shape = (b, m)
        iou_max_masked = tf.cast(tf.greater_equal(iou, max_iou), tf.float32) * iou

        # shape = (b, 1)，无正样本mask。
        # 若某一样本对无大于阈值的正样本ROI，则其在此mask上对应位置上的值为1，否则为0
        no_pos_mask = tf.reduce_sum(pos_mask, axis=1, keepdims=True)
        no_pos_mask = tf.cast(tf.less_equal(no_pos_mask, 0.), tf.float32)

        # shape = (b, N)，通过mask，提取出所有符合要求的ROI的IOU值
        # 保证至少有一个正的IOU值
        # 对于某一样本对（某一行）：当pos mask不全为0时，按mask取出ROI对应的IoU值，否则的话，用最大IoU的ROI进行填充
        iou_masked = iou * pos_mask + iou_max_masked * no_pos_mask

        # 取大于 IOU 阈值的前 k 个ROI作为正样本
        # shape = (b, k)
        top_iou = tf.nn.top_k(iou_masked, k=self.num_pos[self.head_index]).values
        top_iou = tf.reshape(top_iou, shape=(-1, self.num_pos[self.head_index]))

        # shape = (b, 1)
        top_k_iou = top_iou[:, -1]
        top_k_iou = tf.reshape(top_k_iou, shape=(-1, 1))
        top_k_iou = top_k_iou * tf.cast(tf.greater_equal(top_k_iou, self.iou_thresh[self.head_index]), tf.float32)
        top_k_iou = top_k_iou + self.iou_thresh[self.head_index] * tf.cast(tf.equal(top_k_iou, 0.), tf.float32)
        # shape = (b, N)
        pos = tf.cast(tf.greater_equal(iou_masked, top_k_iou), tf.float32)

        # 保证至少有一个正样本
        if self.max_pos and self.head_index == 0:
            top_2_iou = top_iou[:, 1:2]
            max_pos = tf.cast(tf.greater(iou_masked, top_2_iou), tf.float32)
            pos = pos + max_pos
            pos = tf.cast(tf.greater(pos, 0.), tf.float32)

        pos = pos * neg_mask

        neg = tf.py_func(self.random_choice, [overlap, self.head_index], [tf.float32])[0]
        neg = tf.reshape(neg, shape=tf.shape(pos))

        label = tf.ones_like(overlap, dtype=tf.float32)

        label = -1. * label + 1. * neg + 2. * pos
        return label

    def random_choice(self, overlap, head_index):
        label = np.zeros_like(overlap, dtype=np.float32)
        batch = overlap.shape[0]
        for i in range(batch):
            neg = np.where(overlap[i, :] <= self.neg_overlap_thresh[head_index])
            num = neg[0].shape[0]
            if num > self.num_neg[head_index]:
                slt = np.arange(num)
                np.random.shuffle(slt)
                slt = slt[:self.num_neg[head_index]]
                neg = tuple(p[slt] for p in neg)
            label[i, neg[0]] = 1.
        return label


class LabelTriplet(Layer):
    def __init__(self, max_num, pos_overlap_thresh, neg_overlap_thresh, **kwargs):
        super(LabelTriplet, self).__init__(**kwargs)
        self.max_num = max_num
        self.pos_overlap_thresh = pos_overlap_thresh
        self.neg_overlap_thresh = neg_overlap_thresh
        index = tf.range(start=self.max_num, limit=0., delta=-1., dtype=tf.float32)
        self.index = tf.reshape(index, shape=(1, -1))

    def get_config(self):
        base_config = super(LabelTriplet, self).get_config()
        base_config.update({'max_num': self.max_num,
                            'pos_overlap_thresh': self.pos_overlap_thresh,
                            'neg_overlap_thresh': self.neg_overlap_thresh})
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        # shape = (b, m)
        overlap = inputs[0]
        iou = inputs[1]

        iou = iou + 0.01
        label = tf.ones_like(overlap, dtype=tf.float32)

        """
        在 overlap 大于阈值 的 ROI 中筛选正样本
        有符合阈值条件的ROI时，选IOU最高的
        无符合阈值条件的ROI时，选IOU最高的ROI填充
        """
        # shape = (b, m)
        # 有可能出现某一行（某样本对）没有符合阈值要求的ROI的情况，即该行的overlap mask的值均为0
        pos_mask = tf.cast(tf.greater_equal(overlap, self.pos_overlap_thresh), tf.float32)

        # shape = (b, 1)
        max_iou = tf.reduce_max(iou, axis=1, keepdims=True)
        # shape = (b, m)
        iou_max_masked = tf.cast(tf.greater_equal(iou, max_iou), tf.float32) * iou

        # shape = (b, 1)，无正样本mask。
        # 若某一样本对未筛选出正样本ROI，则其在此mask上对应位置上的值为1，否则为0
        no_pos_mask = tf.reduce_sum(pos_mask, axis=1, keepdims=True)
        no_pos_mask = tf.cast(tf.less_equal(no_pos_mask, 0.), tf.float32)

        # shape = (b, m)，通过mask，提取出所有符合要求的ROI的IOU值
        # 对于某一样本对（某一行）：当pos mask不全为0时，按mask取出ROI对应的IoU值，否则的话，用最大IoU的ROI进行填充
        iou_pos_masked = iou * pos_mask + iou_max_masked * no_pos_mask

        # shape = (b, 1)
        max_iou_pos = tf.reduce_max(iou_pos_masked, axis=1, keepdims=True)
        # shape = (b, m)
        pos = tf.cast(tf.greater_equal(iou_pos_masked, max_iou_pos), tf.float32)

        """
        在 overlap 小于阈值 的 ROI 中筛选负样本
        有符合阈值条件的ROI时，尽量选得分高的（最靠前的）
        无符合阈值条件的ROI时，选IOU最低的ROI填充
        """
        neg_mask = tf.cast(tf.less_equal(overlap, self.neg_overlap_thresh), tf.float32)

        # shape = (b, 1)
        min_iou = tf.reduce_min(iou, axis=1, keepdims=True)
        # shape = (b, m)
        iou_min_masked = tf.cast(tf.less_equal(iou, min_iou), tf.float32) * iou

        # shape = (b, 1)，无负样本mask。
        # 若某一样本对未筛选出负样本ROI，则其在此mask上对应位置上的值为1，否则为0
        no_neg_mask = tf.reduce_sum(neg_mask, axis=1, keepdims=True)
        no_neg_mask = tf.cast(tf.less_equal(no_neg_mask, 0.), tf.float32)

        # shape = (b, m)
        iou_neg_masked_ = iou * neg_mask + iou_min_masked * no_neg_mask
        iou_neg_masked = tf.cast(tf.greater(iou_neg_masked_, 0.), tf.float32)

        # shape = (b, m)
        index_masked = iou_neg_masked * self.index
        # shape = (b, 1)
        max_index_masked = tf.reduce_max(index_masked, axis=1, keepdims=True)

        neg = tf.cast(tf.greater_equal(index_masked, max_index_masked), tf.float32)

        label = -1. * label + 1. * neg + 2. * pos
        return label


class EuclidDistance(Layer):
    def __init__(self, **kwargs):
        super(EuclidDistance, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        num_roi = input_shape[1] - 1
        return tuple([batch, num_roi])

    # shape = (b, m + 1, d)
    def call(self, inputs, **kwargs):
        # shape = (b, 1, d)
        zf = inputs[:, 0: 1, :]
        # shape = (b, m, d)
        xf = inputs[:, 1:, :]

        # shape = (b, m, d)
        distance = tf.math.square(xf - zf)
        # shape = (b, m)
        distance = tf.reduce_sum(distance, axis=-1, keepdims=False)
        return distance


class TripletLoss(Layer):
    def __init__(self, num_roi=5, margin=1.0, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        self.num_roi = num_roi
        self.margin = margin

    def get_config(self):
        base_config = super(TripletLoss, self).get_config()
        base_config.update({'num_roi': self.num_roi, 'margin': self.margin})
        return base_config

    def compute_output_shape(self, input_shape):
        return (1, )

    def call(self, inputs, **kwargs):
        # shape = (b, m)
        label = inputs[0]
        # shape = (b, m)
        distance = inputs[1]

        batch = tf.cast(tf.shape(label)[0], dtype=tf.float32)

        # shape = (b, m)
        pos_mask = tf.cast(tf.equal(label, 1.), dtype=tf.float32)
        # shape = (b, m)
        neg_mask = tf.cast(tf.equal(label, 0.), dtype=tf.float32)

        # shape = (b, m)
        pos_distance = pos_mask * distance
        # shape = (b, m)
        neg_distance = neg_mask * distance

        # shape = (b, )
        pos_dis = tf.reduce_sum(pos_distance, axis=-1)
        # shape = (b, )
        neg_dis = tf.reduce_sum(neg_distance, axis=-1)

        # shape = (b, )
        basic_loss = pos_dis - neg_dis + self.margin

        basic_loss = tf.maximum(basic_loss, 0.)

        loss = tf.reduce_sum(basic_loss) / batch
        return loss


class ROIOverlap(Layer):
    def __init__(self, **kwargs):
        super(ROIOverlap, self).__init__(**kwargs)
        pass

    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1], input_shape[0][:-1]]

    def call(self, inputs, **kwargs):
        return list(self.roi_iou(inputs[0], inputs[1]))

    @staticmethod
    def roi_iou(boxes1, boxes2):
        boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                     boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                     boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
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
        roi_overlap = inter_area / (boxes2_area + 1e-9)

        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-9)
        return roi_overlap, iou


class OverlapLoss(Layer):
    def __init__(self, k, **kwargs):
        super(OverlapLoss, self).__init__(**kwargs)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (1, )

    def call(self, inputs, **kwargs):
        overlap = inputs[0]
        mask = inputs[1]

        overlap = tf.maximum(overlap, 1e-5)
        loss = -self.k * tf.log(overlap)
        loss = loss * mask

        num_sample = tf.reduce_sum(tf.cast(tf.greater(mask, 0.), dtype=tf.float32))
        loss = tf.reduce_sum(loss) / (num_sample + 1e-5)
        return loss


class ROISelect(Layer):
    def __init__(self, k, **kwargs):
        super(ROISelect, self).__init__(**kwargs)
        self.k = k

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        return [(batch, self.k, 4), (batch, self.k)]

    def get_config(self):
        base_config = super(ROISelect, self).get_config()
        base_config.update({'k': self.k})
        return base_config

    def call(self, inputs, **kwargs):
        # shape = (b, N)
        score = inputs[0]
        # shape = (b, N, 4)
        roi = inputs[1]

        batch = tf.shape(score)[0]
        batch = tf.cast(batch, dtype=tf.int64)

        # shape = (b, k)
        top_k = tf.nn.top_k(score, k=self.k, sorted=True)
        top_k_score = top_k.values
        top_k_index = top_k.indices
        top_k_index = tf.cast(top_k_index, dtype=tf.int64)
        # shape = (b × k, 1)
        top_k_index = tf.reshape(top_k_index, (-1, 1))

        # shape = (b, )
        batch_index = tf.range(0, batch, dtype=tf.int64)
        # shape = (b, 1)
        batch_index = tf.reshape(batch_index, shape=(-1, 1))
        # shape = (b, k)
        batch_index = tf.tile(batch_index, multiples=(1, self.k))
        # shape = (b × k, 1)
        batch_index = tf.reshape(batch_index, shape=(-1, 1))

        # shape = (b × k, 2)
        roi_index = tf.concat([batch_index, top_k_index], axis=1)

        # shape = (b × k, 4)
        roi_selected = tf.gather_nd(roi, roi_index)
        # shape = (b, k, 4)
        roi_selected = tf.reshape(roi_selected, (-1, self.k, 4))
        return roi_selected, top_k_score


class MaxROI(Layer):
    def __init__(self, max_num=5, iou_thresh=0.5, k=24, score_thresh=0.6, **kwargs):
        super(MaxROI, self).__init__(**kwargs)
        self.max_num = max_num
        self.iou_thresh = iou_thresh
        self.k = k
        self.score_thresh = score_thresh

    def get_config(self):
        base_config = super(MaxROI, self).get_config()
        base_config.update({'max_num': self.max_num, 'iou_thresh': self.iou_thresh, 'k': self.k, 'score_thresh': self.score_thresh})
        return base_config

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        return tuple([batch, self.max_num, 4])

    def _call(self, inputs, **kwargs):
        # shape = (b, N, 4), (b, N, 2)
        boxes_batch, scores_batch = inputs[0], inputs[1]
        batch = boxes_batch.shape[0].value

        scores_batch = tf.nn.softmax(scores_batch, axis=-1)
        # shape = (b, N)
        scores_batch = scores_batch[..., 1]

        rois = []

        """
        共N个box，其中M个满足得分要求
        第一次筛选中有m1个满足iou要求
        第二次从剩余的M-m1中筛选，又有m2个满足要求，
        第三次从剩余的M-m1-m2中筛选，又有m3个满足要求
        ...
        第max_box次时...
        """
        for i in range(batch):
            # shape = (N, 4)
            boxes = boxes_batch[i, ...]
            # shape = (N, )
            scores = scores_batch[i, :]

            # 负样本保底措施
            neg_index = tf.where(tf.less(scores, self.score_thresh))[:, 0]
            neg_scores = tf.gather(scores, neg_index)
            neg_boxes = tf.gather(boxes, neg_index)
            top_1_index = tf.nn.top_k(neg_scores, k=1).indices
            top_1_index = tf.cast(top_1_index, dtype=tf.int64)
            neg_box = tf.gather(neg_boxes, top_1_index)

            # 保底措施，无高于得分阈值或满足iou条件的box时，用top1的ROI填充占位
            # shape = (1, )
            top_1_index = tf.nn.top_k(scores, k=1).indices
            top_1_index = tf.cast(top_1_index, dtype=tf.int64)

            # 将不满足得分阈值的box筛去，共得到M个box，M可能为0
            # shape = (M, )
            pos_index_ = tf.where(tf.greater_equal(scores, self.score_thresh))[:, 0]

            # 无满足得分条件的box时，用top1得分box填充，确保至少有一个box进入后续筛选
            pos_index = tf.cond(tf.equal(tf.size(pos_index_), 0), lambda: top_1_index, lambda: pos_index_)

            # shape = (M, ), M ≥ 1
            pos_scores = tf.gather(scores, pos_index)
            # shape = (M, 4), M ≥ 1
            pos_boxes = tf.gather(boxes, pos_index)

            roi_ = []
            for j in range(self.max_num - 1):
                # shape = (1, )
                max_index = tf.nn.top_k(pos_scores, k=1).indices
                # shape = (1, )
                max_score = tf.gather(pos_scores, max_index)
                # shape = (1, 4)
                max_box = tf.gather(pos_boxes, max_index)

                # 计算当前最高得分box与剩余所有box(包括自身)的iou
                # shape = (M, )
                iou = bbox_iou(max_box, pos_boxes, manner='corner')[0]

                # 筛选出m1个满足条件的box，m1 ≥ 1（因为包括自身）
                # shape = (m1, )
                over_index = tf.where(tf.greater_equal(iou, self.iou_thresh))[:, 0]
                # shape = (m1, 4)
                over_boxes = tf.gather(pos_boxes, over_index)

                # 确定出目标可能存在的最大范围
                # shape = (1, 2)
                roi_x1y1 = tf.reduce_min(over_boxes[:, :2], axis=0, keepdims=True)
                # shape = (1, 2)
                roi_x2y2 = tf.reduce_max(over_boxes[:, 2:], axis=0, keepdims=True)
                # shape = (1, 4)
                roi = tf.concat([roi_x1y1, roi_x2y2], axis=-1)

                # 该次筛选完成
                roi_.append(roi)

                # 整理被筛选掉的box，准备进行下次筛选
                # shape = (M - m1, )，有可能为0
                next_index = tf.where(tf.less(iou, self.iou_thresh))[:, 0]

                # shape = (M - m1, )
                pos_scores_ = tf.gather(pos_scores, next_index)
                # shape = (M - m1, 4)
                pos_boxes_ = tf.gather(pos_boxes, next_index)

                # 保底措施，保证至少为1
                pos_scores = tf.cond(tf.equal(tf.size(next_index), 0), lambda: max_score, lambda: pos_scores_)
                pos_boxes = tf.cond(tf.equal(tf.size(next_index), 0), lambda: roi, lambda: pos_boxes_)

            roi_.append(neg_box)
            # shape = (max_num, 4)
            roi_ = tf.concat(roi_, axis=0)
            rois.append(roi_)

        # shape = (b, max_num, 4)
        rois = tf.stack(rois, axis=0)
        return rois

    def call(self, inputs, **kwargs):
        # shape = (b, N, 4), (b, N, 2)
        boxes_batch, scores_batch = inputs[0], inputs[1]
        batch = boxes_batch.shape[0].value

        scores_batch = tf.nn.softmax(scores_batch, axis=-1)
        # shape = (b, N)
        scores_batch = scores_batch[..., 1]

        rois = []

        k = self.k
        """
        共N个box，其中M个满足得分要求
        第一次筛选中有m1个满足iou要求
        第二次从剩余的M-m1中筛选，又有m2个满足要求，
        第三次从剩余的M-m1-m2中筛选，又有m3个满足要求
        ...
        第max_box次时...
        """
        for i in range(batch):
            # shape = (N, 4)
            boxes = boxes_batch[i, ...]
            # shape = (N, )
            scores = scores_batch[i, :]

            # shape = (k + m, )
            top_k_index = tf.nn.top_k(scores, k=(k + self.max_num)).indices
            top_k_index = tf.cast(top_k_index, dtype=tf.int64)

            # shape = (k + m, )
            score_ = tf.gather(scores, top_k_index)
            # shape = (k + m, 4)
            box_ = tf.gather(boxes, top_k_index)

            # shape = (k, )
            score = score_[:k]
            # shape = (k, 4)
            box = box_[:k, :]

            roi_ = []
            for j in range(self.max_num - 1):
                max_box = box[0:1, :]

                # 前k个Box都筛选完后，占位
                placeholder_box = box_[k + j: k + j + 1, :]
                placeholder_score = score_[k + j: k + j + 1]

                # 计算当前最高得分box与剩余所有box(包括自身)的iou
                # shape = (m0, )
                iou = bbox_iou(max_box, box, manner='corner')[0]

                # 筛选出m1个满足条件的box，m1 ≥ 1（因为包括自身）
                # shape = (m1, )
                over_index = tf.where(tf.greater_equal(iou, self.iou_thresh))[:, 0]
                # shape = (m1, 4)
                over_box = tf.gather(box, over_index)

                # 确定出目标可能存在的最大范围
                # shape = (1, 2)
                roi_x1y1 = tf.reduce_min(over_box[:, :2], axis=0, keepdims=True)
                # shape = (1, 2)
                roi_x2y2 = tf.reduce_max(over_box[:, 2:], axis=0, keepdims=True)
                # shape = (1, 4)
                roi = tf.concat([roi_x1y1, roi_x2y2], axis=-1)

                # 该次筛选完成
                roi_.append(roi)

                # 整理被筛选掉的box，准备进行下次筛选
                # shape = (m0 - m1, )，有可能为0
                next_index = tf.where(tf.less(iou, self.iou_thresh))[:, 0]

                # shape = (m0 - m1, )
                score0 = tf.gather(score, next_index)
                # shape = (m0 - m1, 4)
                box0 = tf.gather(box, next_index)

                # 保底措施，保证至少有1个box进入下次筛选
                score = tf.cond(tf.equal(tf.size(next_index), 0), lambda: placeholder_score, lambda: score0)
                box = tf.cond(tf.equal(tf.size(next_index), 0), lambda: placeholder_box, lambda: box0)

            # 保底措施，保证至少有1个满足负样本条件的box
            roi_.append(box_[-2:-1, :])
            # shape = (max_num, 4)
            roi_ = tf.concat(roi_, axis=0)
            rois.append(roi_)

        # shape = (b, max_num, 4)
        rois = tf.stack(rois, axis=0)
        return rois


class ROIGeneration(Layer):
    def __init__(self, num_local_roi, num_global_roi, global_roi_level, extension, max_size_ratio, min_size_ratio, **kwargs):
        super(ROIGeneration, self).__init__(**kwargs)
        self.num_local_roi = num_local_roi
        self.num_global_roi = num_global_roi
        self.global_roi_level = global_roi_level
        self.extension = extension
        self.max_size_ratio = max_size_ratio
        self.min_size_ratio = min_size_ratio
        self.global_roi = tf.constant(self.init_global_roi(), dtype=tf.float32)

    def get_config(self):
        base_config = super(ROIGeneration, self).get_config()
        base_config.update({'num_local_roi': self.num_local_roi, 'num_global_roi': self.num_global_roi,
                            'global_roi_level': self.global_roi_level, 'extension': self.extension,
                            'min_size_ratio': self.min_size_ratio, 'max_size_ratio': self.max_size_ratio})
        return base_config

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        return (batch, self.num_local_roi + self.num_global_roi, 4)

    # def build(self, input_shape):
    #     self.global_roi = self.add_weight(name='global_roi',
    #                                       shape=(self.num_global_roi, 4),
    #                                       initializer=constant(self.init_global_roi()))

    def call(self, inputs, **kwargs):
        batch = tf.shape(inputs)[0]
        global_roi = tf.tile(self.global_roi[None, ...], multiples=(batch, 1, 1))
        local_roi = self.init_local_roi(inputs, batch)
        roi = tf.concat([global_roi, local_roi], axis=-2)
        return roi

    def init_global_roi(self):
        boxes = []
        roi_level = self.global_roi_level
        size = 1. / roi_level
        for i in range(roi_level):
            grid = np.arange(0., 0.999 - i * size, size)
            y = np.reshape(np.tile(grid, roi_level - i), (-1, 1))
            x = np.reshape(np.tile(np.reshape(grid, (-1, 1)), (1, roi_level - i)), (-1, 1))
            box = np.concatenate([x, y, x + size * (i + 1), y + size * (i + 1)], axis=-1)
            boxes.append(box)
        boxes = np.concatenate(boxes)
        return boxes

    def init_local_roi(self, z_box, batch):
        wh = z_box[:, 2:] - z_box[:, :2]
        xy = (z_box[:, 2:] + z_box[:, :2]) / 2.

        center = tf.random_normal(shape=(batch, self.num_local_roi, 2), mean=0., stddev=(self.extension - 1.) / (2. * 3))
        size = tf.random_uniform(shape=(batch, self.num_local_roi, 2), minval=self.min_size_ratio, maxval=self.max_size_ratio)

        trans_xy = xy[:, None, :] + wh[:, None, :] * center
        scale_wh = wh[:, None, :] * size

        x1y1 = trans_xy - scale_wh / 2.
        x2y2 = trans_xy + scale_wh / 2.

        left_top = tf.abs(x1y1) * tf.cast(tf.less(x1y1, 0.), dtype=tf.float32)
        right_bottom = (x2y2 - 1.) * tf.cast(tf.greater(x2y2, 1.), dtype=tf.float32)

        x1y1 = x1y1 + left_top - right_bottom
        x1y1 = tf.maximum(x1y1, 0.)
        x2y2 = x2y2 + left_top - right_bottom
        x2y2 = tf.minimum(x2y2, 1.)

        global_roi = tf.concat([x1y1, x2y2], axis=-1)
        return global_roi


class ChannelMaxPooling(Layer):
    def __init__(self, out_planes, **kwargs):
        super(ChannelMaxPooling, self).__init__(**kwargs)
        self.out_planes = out_planes

    def get_config(self):
        base_config = super(ChannelMaxPooling, self).get_config()
        base_config.update({'out_planes': self.out_planes})
        return base_config

    def call(self, inputs, **kwargs):
        batch = inputs.shape[0].value
        size = inputs.shape[1].value

        channel_per_pixel = self.out_planes // size ** 2
        channel_center_pixel = self.out_planes % size ** 2

        out = tf.nn.top_k(inputs, k=channel_per_pixel).values
        out = tf.reshape(out, (batch, -1))
        center = tf.nn.top_k(inputs[:, size // 2, size // 2, :], k=channel_center_pixel).values
        out = tf.concat([center, out], axis=-1)
        return out

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        return tuple([input_shape[0], 1, 1, self.out_planes])


class SpatialGlobalAvgPool(Layer):
    def __init__(self, **kwargs):
        super(SpatialGlobalAvgPool, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    def call(self, inputs, **kwargs):
        xf = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        return xf


class SpatialGlobalMaxPool(Layer):
    def __init__(self, **kwargs):
        super(SpatialGlobalMaxPool, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    def call(self, inputs, **kwargs):
        xf = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return xf
