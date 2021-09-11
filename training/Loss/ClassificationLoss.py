

import tensorflow as tf
from keras.layers import Layer


class PreprocessCls(Layer):
    def __init__(self, mode, **kwargs):
        super(PreprocessCls, self).__init__(**kwargs)
        self.mode = mode

    def compute_output_shape(self, input_shape):
        s1 = list(input_shape[0])
        s1_ = 1
        for s in s1:
            s1_ = s1_ * s
        s2 = list(input_shape[1])
        s2_ = 1
        for s in s2:
            s2_ = s2_ * s
        return [(s2_, ), (s1_ // 2, 2)]

    def get_config(self):
        base_config = super(PreprocessCls, self).get_config()
        base_config.update({'mode': self.mode})
        return base_config

    def call(self, inputs, **kwargs):
        label, pred = inputs

        if self.mode == 'log-softmax':
            pred = tf.nn.log_softmax(pred, axis=-1)
        elif self.mode == 'sigmoid':
            pred = tf.sigmoid(pred)
        elif self.mode == 'softmax':
            pred = tf.nn.softmax(pred, axis=-1)

        label = tf.reshape(label, shape=(-1,))
        pred = tf.reshape(pred, shape=(-1, 2))
        return [label, pred]


def nll_loss(pred, label):
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), 2, dtype=pred.dtype)
    loss = -label * pred
    return loss


class NllPosLoss(Layer):
    def __init__(self, **kwargs):
        super(NllPosLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1, )

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]

        pos_index = tf.where_v2(tf.equal(label, 1.))[:, 0]
        pos_label = tf.gather(label, pos_index)
        num_pos = tf.reduce_sum(tf.ones_like(pos_label, dtype=tf.float32))
        pos_pred = tf.gather(pred, pos_index)
        pos_loss_ = nll_loss(pos_pred, pos_label)[:, 1]

        if len(inputs) == 3:
            weights = inputs[2]
            weights = tf.reshape(weights, shape=(-1,))
            pos_weights = tf.gather(weights, pos_index)
            pos_loss_ = pos_loss_ * pos_weights

        pos_loss = tf.reduce_sum(pos_loss_) / (num_pos + 1e-6)
        return pos_loss


class NllNegLoss(Layer):
    def __init__(self, **kwargs):
        super(NllNegLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1, )

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]

        neg_index = tf.where_v2(tf.equal(label, 0.))[:, 0]
        neg_label = tf.gather(label, neg_index)
        num_neg = tf.reduce_sum(tf.ones_like(neg_label, dtype=tf.float32))
        neg_pred = tf.gather(pred, neg_index)

        neg_loss = nll_loss(neg_pred, neg_label)[:, 0]
        neg_loss = tf.reduce_sum(neg_loss) / (num_neg + 1e-6)
        return neg_loss


class FocalNegLoss(Layer):
    def __init__(self, alpha=0.75, gamma=2.0, **kwargs):
        super(FocalNegLoss, self).__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (1, )

    def get_config(self):
        base_config = super(FocalNegLoss, self).get_config()
        base_config.update({'alpha': self.alpha, 'gamma': self.gamma})
        return base_config

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]

        neg_index = tf.where_v2(tf.equal(label, 0.))[:, 0]
        neg_label = tf.gather(label, neg_index)
        num_neg = tf.reduce_sum(tf.ones_like(neg_label, dtype=tf.float32))
        neg_pred = tf.gather(pred, neg_index)[:, 0]

        neg_loss_ = - self.alpha * tf.math.pow((1. - neg_pred), self.gamma) * tf.log(tf.maximum(neg_pred, 0.0001))
        neg_loss = tf.reduce_sum(neg_loss_) / (num_neg + 1e-6)
        return neg_loss


class FocalPosLoss(Layer):
    def __init__(self, alpha=.25, gamma=2.0, **kwargs):
        super(FocalPosLoss, self).__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (1, )

    def get_config(self):
        base_config = super(FocalPosLoss, self).get_config()
        base_config.update({'alpha': self.alpha, 'gamma': self.gamma})
        return base_config

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]

        pos_index = tf.where_v2(tf.equal(label, 1.))[:, 0]
        pos_label = tf.gather(label, pos_index)
        num_pos = tf.reduce_sum(pos_label)
        pos_pred = tf.gather(pred, pos_index)[:, 1]

        pos_loss = -self.alpha * tf.math.pow((1. - pos_pred), self.gamma) * tf.log(tf.maximum(pos_pred, 1e-6))
        pos_loss = tf.reduce_sum(pos_loss) / (num_pos + 1e-6)
        # pos_loss = tf.reduce_sum(pos_loss) / (tf.cast(tf.shape(pos_loss), tf.float32) + 1e-6)
        return pos_loss
