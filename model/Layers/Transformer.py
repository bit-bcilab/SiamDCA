

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Input, Dense, ReLU, Dropout, Lambda, Reshape, Add, Dot
from keras.initializers import he_normal

from .operation import Transpose, MatMul
from .activation import Softmax
from .convolution import LayerNormalization


def ScaledDotProductAttention(num_heads, seq_length, d_k, d_v, dropout):
    q = Input(shape=(num_heads, seq_length, d_k))
    k = Input(shape=(num_heads, seq_length, d_k))
    v = Input(shape=(num_heads, seq_length, d_v))

    k_ = Transpose(perm=(0, 1, 3, 2))(k)
    matmul_qk = MatMul()([q, k_])
    scaled_attention_logits = Lambda(lambda tensor: tensor / np.sqrt(d_k))(matmul_qk)

    attention_weights = Softmax()(scaled_attention_logits)
    attention_weights = Dropout(rate=dropout)(attention_weights)

    output = MatMul()([attention_weights, v])
    return Model(inputs=[q, k, v], outputs=[output, attention_weights])


def MultiHeadAttention(num_heads, model_dims, seq_length, d_k, d_v, dropout=.1, use_pe=False, use_bias=False):
    k_dims = int(num_heads * d_k)
    v_dims = int(num_heads * d_v)
    q_ = Input(shape=(seq_length, model_dims))
    k_ = Input(shape=(seq_length, model_dims))
    v_ = Input(shape=(seq_length, model_dims))
    att = ScaledDotProductAttention(num_heads, seq_length, d_k, d_v, dropout=dropout)

    if use_pe:
        q_pe = PositionalEncoding(seq_length=seq_length, model_dims=model_dims)(q_)
        k_pe = PositionalEncoding(seq_length=seq_length, model_dims=model_dims)(k_)
        q = Dense(units=k_dims, input_dim=model_dims, use_bias=use_bias, kernel_initializer=he_normal())(q_pe)
        k = Dense(units=k_dims, input_dim=model_dims, use_bias=use_bias, kernel_initializer=he_normal())(k_pe)
    else:
        q = Dense(units=k_dims, input_dim=model_dims, use_bias=use_bias, kernel_initializer=he_normal())(q_)
        k = Dense(units=k_dims, input_dim=model_dims, use_bias=use_bias, kernel_initializer=he_normal())(k_)

    q = Reshape(target_shape=(seq_length, num_heads, d_k))(q)
    q = Transpose(perm=(0, 2, 1, 3))(q)

    k = Reshape(target_shape=(seq_length, num_heads, d_k))(k)
    k = Transpose(perm=(0, 2, 1, 3))(k)

    v = Dense(units=v_dims, input_dim=model_dims, use_bias=use_bias, kernel_initializer=he_normal())(v_)
    v = Reshape(target_shape=(seq_length, num_heads, d_v))(v)
    v = Transpose(perm=(0, 2, 1, 3))(v)

    outputs, attention = att([q, k, v])
    outputs = Transpose(perm=(0, 2, 1, 3))(outputs)
    outputs = Reshape(target_shape=(seq_length, v_dims))(outputs)

    outputs = Dense(units=model_dims, input_dim=v_dims, use_bias=use_bias, kernel_initializer=he_normal())(outputs)
    if use_pe:
        outputs = Add()([outputs, q_pe])
    else:
        outputs = Add()([outputs, q_])
    outputs = LayerNormalization(epsilon=1e-6)(outputs)
    return Model(inputs=[q_, k_, v_], outputs=[outputs, attention])


def FeedForwardNet(seq_length, model_dims, hidden_dims, dropout):
    inputs = Input(shape=(seq_length, model_dims))
    outputs = Dense(units=hidden_dims, input_dim=model_dims, kernel_initializer=he_normal())(inputs)
    outputs = ReLU()(outputs)
    outputs = Dense(units=model_dims, input_dim=hidden_dims, kernel_initializer=he_normal())(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    outputs = Add()([outputs, inputs])
    outputs = LayerNormalization(epsilon=1e-6)(outputs)
    return Model(inputs=inputs, outputs=outputs)


class PositionalEncoding(Layer):
    def __init__(self, seq_length, model_dims, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.seq_length = seq_length
        self.model_dims = model_dims
        self.pos_encoding = tf.constant(self.PositionalEncoding(), dtype=tf.float32)

    def get_config(self):
        base_config = super(PositionalEncoding, self).get_config()
        base_config.update({'seq_length': self.seq_length, 'model_dims': self.model_dims})
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        batch = tf.shape(inputs)[0]
        pos_encoding = tf.tile(self.pos_encoding[None, ...], multiples=(batch, 1, 1))
        outputs = inputs + pos_encoding
        return outputs

    @staticmethod
    def get_angles(pos, i, model_dims):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dims))
        return pos * angle_rates

    def PositionalEncoding(self):
        angle_rads = self.get_angles(np.arange(self.seq_length)[:, None],
                                     np.arange(self.model_dims)[None, :],
                                     self.model_dims)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        cones = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.stack([sines, cones], axis=-1)
        pos_encoding = np.reshape(pos_encoding, (self.seq_length, -1))
        return pos_encoding
