

import tensorflow as tf
from keras.layers import Layer


def anchor_based_decoder(cls, delta, anchors, score_size, anchor_num, act='softmax', flatten=True):
    cls = tf.reshape(cls, shape=(score_size, score_size, anchor_num, 2))
    if act == 'softmax':
        score = tf.nn.softmax(cls, axis=-1)[..., 1]  # nll loss
    else:
        score = tf.sigmoid(cls)  # focal loss

    delta = tf.reshape(delta, shape=(score_size, score_size, anchor_num, 4))
    box_xy = delta[..., :2] * anchors[..., 2:] + anchors[..., :2]
    box_wh = tf.exp(delta[..., 2:]) * anchors[..., 2:]
    bbox = tf.concat([box_xy, box_wh], axis=-1)

    if flatten:
        score = tf.reshape(score, shape=(-1, ))
        bbox = tf.reshape(bbox, shape=(-1, 4))
    return score, bbox


class AnchorBasedDecoder(Layer):
    def __init__(self, anchors, **kwargs):
        super(AnchorBasedDecoder, self).__init__(**kwargs)
        self.anchors = anchors

    def compute_output_shape(self, input_shape):
        s1 = input_shape[0]
        return [s1, s1]

    def get_config(self):
        base_config = super(AnchorBasedDecoder, self).get_config()
        base_config.update({'anchors': self.anchors})
        return base_config

    def call(self, inputs, **kwargs):
        label = inputs[0]
        pred = inputs[1]
        shape = tf.shape(label)

        pred = tf.reshape(pred, shape=shape)
        pred_box = self.decode(pred)
        bbox = self.decode(label)
        return [bbox, pred_box]

    def decode(self, inputs):
        box_xy = inputs[..., :2] * self.anchors[..., 2:] + self.anchors[..., :2]
        box_wh = tf.exp(inputs[..., 2:]) * self.anchors[..., 2:]
        box = tf.concat([box_xy, box_wh], axis=-1)
        return box
