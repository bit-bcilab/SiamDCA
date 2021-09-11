

import tensorflow as tf
from keras.layers import Layer


def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)


def swish(x):
    return tf.nn.swish(x)


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


def sigmoid(x):
    return tf.nn.sigmoid(x)


def leaky(x):
    return tf.nn.leaky_relu(x)


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **kwargs):
        return x * tf.nn.tanh(tf.nn.softplus(x))


class Swish(Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **kwargs):
        return tf.nn.swish(x)


class Sigmoid(Layer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **kwargs):
        return tf.sigmoid(x)


class Softmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(Softmax, self).get_config()
        base_config.update({'axis': self.axis})
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **kwargs):
        return tf.nn.softmax(x, axis=self.axis)
