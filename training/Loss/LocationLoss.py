

import tensorflow as tf
from keras.layers import Layer

from utils.iou import bbox_iou, bbox_giou, bbox_diou, bbox_ciou


class L1Loss(Layer):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1, )

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]
        shape = tf.shape(label)
        batch = tf.cast(shape[0], dtype=tf.float32)
        pred = tf.reshape(pred, shape=shape)

        diff = tf.abs(pred - label)
        loss = tf.reduce_sum(diff, axis=-1)

        if len(inputs) > 2:
            mask = inputs[2]
            loss = mask * loss
            num_sample = tf.reduce_sum(tf.cast(tf.greater(mask, 0.), dtype=tf.float32))
            # num_sample = tf.reduce_sum(mask)
            loss = tf.reduce_sum(loss) / (num_sample + 1e-6)
        else:
            loss = tf.reduce_sum(loss) / batch
        return loss


class SmoothL1Loss(Layer):
    def __init__(self, sigma=3., **kwargs):
        super(SmoothL1Loss, self).__init__(**kwargs)
        self.sigma_squared = tf.constant(sigma ** 2, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (1, )

    def get_config(self):
        base_config = super(SmoothL1Loss, self).get_config()
        base_config.update({'sigma_squared': self.sigma_squared})
        return base_config

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]

        shape = tf.shape(label)
        batch = tf.cast(shape[0], dtype=tf.float32)
        pred = tf.reshape(pred, shape=shape)

        diff = tf.abs(pred - label)

        loss = tf.where_v2(tf.less(diff,  1. / self.sigma_squared),
                           0.5 * self.sigma_squared * tf.pow(diff, 2.),
                           diff - 0.5 / self.sigma_squared)
        loss = tf.reduce_sum(loss, axis=-1)

        if len(inputs) > 2:
            mask = inputs[2]
            loss = mask * loss
            num_sample = tf.reduce_sum(tf.cast(tf.greater(mask, 0.), dtype=tf.float32))
            # num_sample = tf.reduce_sum(mask)
            loss = tf.reduce_sum(loss) / (num_sample + 1e-6)
        else:
            loss = tf.reduce_sum(loss) / batch
        return loss


class IOULoss(Layer):
    def __init__(self, return_iou=False, **kwargs):
        super(IOULoss, self).__init__(**kwargs)
        self.return_iou = return_iou

    def get_config(self):
        base_config = super(IOULoss, self).get_config()
        base_config.update({'return_iou': self.return_iou})
        return base_config

    def compute_output_shape(self, input_shape):
        return [(1, ), input_shape[0][:-1]]

    def call(self, inputs, **kwargs):
        bbox = inputs[0]
        pred_box = inputs[1]
        batch = tf.cast(tf.shape(bbox)[0], dtype=tf.float32)

        iou_, iou = self.iou_func(bbox, pred_box)
        iou_loss = 1. - iou_

        if len(inputs) > 2:
            mask = inputs[2]
            iou_loss = mask * iou_loss
            num_sample = tf.reduce_sum(tf.cast(tf.greater(mask, 0.), dtype=tf.float32))
            # num_sample = tf.reduce_sum(mask)
            iou_loss = tf.reduce_sum(iou_loss) / (num_sample + 1e-6)
        else:
            iou_loss = tf.reduce_sum(iou_loss) / batch

        if self.return_iou:
            return [iou_loss, iou]
        else:
            return [iou_loss, iou_]

    def iou_func(self, bbox, pred_box):
        iou, _, _, _ = bbox_iou(bbox, pred_box)
        return iou, iou


class GIOULoss(IOULoss):
    def __init__(self, **kwargs):
        super(GIOULoss, self).__init__(**kwargs)

    def iou_func(self, bbox, pred_box):
        giou, iou = bbox_giou(bbox, pred_box)
        return giou, iou


class DIOULoss(IOULoss):
    def __init__(self, **kwargs):
        super(DIOULoss, self).__init__(**kwargs)

    def iou_func(self, bbox, pred_box):
        diou, iou = bbox_diou(bbox, pred_box)
        return diou, iou


class CIOULoss(IOULoss):
    def __init__(self, **kwargs):
        super(CIOULoss, self).__init__(**kwargs)

    def iou_func(self, bbox, pred_box):
        ciou, iou = bbox_ciou(bbox, pred_box)
        return ciou, iou
