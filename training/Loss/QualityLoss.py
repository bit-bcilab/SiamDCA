

import tensorflow as tf
from keras.layers import Layer


class ShrinkageLoss(Layer):
    def __init__(self, enhanced, a=10., c=0.2, w1=2.5, w2=2., w3=1., **kwargs):
        super(ShrinkageLoss, self).__init__(**kwargs)
        self.enhanced = enhanced
        self.a = tf.constant(a, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)
        self.w1 = tf.constant(w1, dtype=tf.float32)
        self.w2 = tf.constant(w2, dtype=tf.float32)
        self.w3 = tf.constant(w3, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (1, )

    def get_config(self):
        base_config = super(ShrinkageLoss, self).get_config()
        base_config.update({'enhanced': self.enhanced, 'a': self.a, 'c': self.c, 'w1': self.w1, 'w2': self.w2, 'w3': self.w3})
        return base_config

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]
        batch = tf.cast(tf.shape(pred)[0], dtype=tf.float32)
        label = tf.reshape(label, shape=(-1,))
        pred = tf.reshape(pred, (-1,))

        loss = tf.losses.huber_loss(label, pred, reduction='none')
        # shrinkage_loss = tf.exp(label) * loss / (1. + tf.exp(self.a * (self.c - loss)))
        shrinkage_loss = loss / (1. + tf.exp(self.a * (self.c - loss)))
        if self.enhanced:
            shrinkage_loss = (self.w1 * tf.pow(label, self.a) + self.w2 * label + self.w3) * shrinkage_loss
        shrinkage_loss = tf.reduce_sum(shrinkage_loss) / batch
        return shrinkage_loss


class HuberLoss(Layer):
    def __init__(self, **kwargs):
        super(HuberLoss, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (1, )

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]
        mask = inputs[2]

        batch = tf.cast(tf.shape(pred)[0], dtype=tf.float32)
        # label = tf.reshape(label, shape=(-1, ))
        # pred = tf.reshape(pred, shape=(-1, ))

        loss = tf.losses.huber_loss(label, pred, reduction='none')
        if len(inputs) > 2:
            mask = tf.reshape(mask, shape=(batch, -1))
            mask = tf.reduce_sum(mask, axis=-1, keep_dims=False)
            mask = tf.cast(tf.greater(mask, 0.), dtype=tf.float32)
            loss = loss * mask[:, None, None]

        loss = tf.reduce_sum(loss) / batch
        return loss
