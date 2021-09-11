

from keras.initializers import RandomNormal
from keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, Activation, ReLU, LeakyReLU
from keras.layers import MaxPooling2D, Concatenate, ZeroPadding2D

from .convolution import DCNv2, DepthWiseCorr, LayerNormalization, InstanceNormalization
from .activation import Mish, Sigmoid, Swish, Softmax
from .activation import relu, relu6, mish, sigmoid, swish, leaky
from .operation import MatMul, Squeeze, Split, Stack, Tile, Transpose
from .operation import Center2Corner, Corner2Center, ROIAlign, Box2ROI, WeightedAdd, UpSamplingAdd, UpSamplingConcat
from .Transformer import ScaledDotProductAttention, MultiHeadAttention, FeedForwardNet, PositionalEncoding


def spp(x):
    x_1 = x
    x_2 = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
    x_3 = MaxPooling2D(pool_size=9, strides=1, padding='same')(x)
    x_4 = MaxPooling2D(pool_size=13, strides=1, padding='same')(x)
    out = Concatenate()([x_4, x_3, x_2, x_1])
    return out


def convolution(conv):
    if conv == 'conv':
        fn = Conv2D
    elif conv == 'separable':
        fn = SeparableConv2D
    elif conv == 'depth-wise':
        fn = DepthwiseConv2D
    elif conv == 'deformable':
        fn = DCNv2
    return fn


def get_activation(act):
    if act == 'relu':
        fn = relu
    elif act == 'mish':
        fn = mish
    elif act == 'sigmoid':
        fn = sigmoid
    elif act == 'swish':
        fn = swish
    elif act == 'leaky':
        fn = leaky
    elif act == 'relu6':
        fn = relu6
    return fn


def activation(act):
    if act == 'relu':
        fn = ReLU()
    elif act == 'mish':
        fn = Mish()
    elif act == 'sigmoid':
        fn = Sigmoid()
    elif act == 'swish':
        fn = Swish()
    elif act == 'leaky':
        fn = LeakyReLU()
    return fn


def Unit(inputs, conv='conv', act=None, pad=None, name=None, **kwargs):
    if pad is not None:
        inputs = ZeroPadding2D(pad)(inputs)

    if name is None:
        output = convolution(conv)(**kwargs)(inputs)
        output = BatchNormalization(gamma_initializer=RandomNormal(mean=1., stddev=0.02),
                                    beta_initializer=RandomNormal(mean=0., stddev=0.02))(output)
    else:
        kwargs.update(dict(name=name + '_conv'))
        output = convolution(conv)(**kwargs)(inputs)
        output = BatchNormalization(gamma_initializer=RandomNormal(mean=1., stddev=0.02),
                                    beta_initializer=RandomNormal(mean=0., stddev=0.02),
                                    name=(name + '_bn'))(output)

    if act is not None:
        output = Activation(get_activation(act))(output)
        # output = activation(act)(output)
    return output
