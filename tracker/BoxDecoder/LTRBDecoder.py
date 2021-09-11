

import tensorflow as tf
from keras.layers import Layer


def ltrb_decoder(cls, distance, grid, act='softmax', flatten=False):
    if act == 'softmax':
        score = tf.nn.softmax(cls, axis=-1)[..., 1]  # nll loss
    else:
        score = tf.sigmoid(cls)  # focal loss

    bbox = LTRBDecoder(grid=grid)(distance)

    if flatten:
        score = tf.reshape(score, shape=(-1, ))
        bbox = tf.reshape(bbox, shape=(-1, 4))
    return score, bbox


class LTRBDecoder(Layer):
    def __init__(self, grid, **kwargs):
        super(LTRBDecoder, self).__init__(**kwargs)
        self.grid = grid[None, ...]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(LTRBDecoder, self).get_config()
        base_config.update({'grid': self.grid})
        return base_config

    def call(self, inputs, **kwargs):
        box_x1y1 = self.grid - inputs[..., :2]
        box_x2y2 = self.grid + inputs[..., 2:]
        box = tf.concat([box_x1y1, box_x2y2], axis=-1)
        return box
