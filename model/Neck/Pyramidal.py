

from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate
from keras.initializers import he_normal, VarianceScaling

from model.Backbone import BACKBONE_OUT_FILTERS, FPN_INPUT_SIZE
from model.Layers import Unit, activation, UpSamplingAdd, WeightedAdd, DCNv2


def UpUnit(small_input, big_input, act=None, **kwargs):
    small_input = UpSampling2D(interpolation='bilinear')(small_input)
    output = WeightedAdd(layer_num=2)([small_input, big_input])
    if act is not None:
        output = activation(act)(output)
    output = Unit(output, conv='conv', **kwargs)
    return output


def DownUnit(skip_connect, small_input, big_input, act=None, **kwargs):
    down_input = MaxPooling2D(pool_size=2, strides=2, padding='valid')(big_input)
    if skip_connect is not None:
        output = WeightedAdd(layer_num=3)([skip_connect, small_input, down_input])
    else:
        output = WeightedAdd(layer_num=2)([small_input, down_input])
    if act is not None:
        output = activation(act)(output)
    output = Unit(output, conv='conv', **kwargs)
    return output


def BiFPNBlock(layers, num_filters, idx, num_outputs):
    conv_kwargs = dict(filters=num_filters,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       padding='same',
                       use_bias=True,
                       kernel_initializer=he_normal(seed=None))

    layer_num = len(layers)
    layers_td = [layers[-1]]
    for i in range(layer_num - 1):
        layer_td = UpUnit(small_input=layers_td[i], big_input=layers[layer_num-2-i], act='swish', **conv_kwargs)
        layers_td.append(layer_td)
    layers_td = layers_td[::-1]

    # 第一个Block里，用于跳连的输入层需要额外处理一次
    if idx == 0:
        inputs_kwargs = dict(filters=num_filters,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='valid',
                             use_bias=False,
                             kernel_initializer=he_normal(seed=None))
        for i in range(1, layer_num - 1):
            layers[i] = Unit(layers[i], conv='conv', **inputs_kwargs)

    # 如果只输出最大尺寸层，那么最后一个Block无需再进行下采样
    if num_outputs == 1 and idx == -1:
        return layers_td
    else:
        # 下采样循环
        layers_out = [layers_td[0]]  # 上采样得到的最大尺寸层直接为输出层
        for i in range(1, layer_num):
            # 最后一层输出没有跳连
            if i == layer_num - 1:
                skip_connect = None
            else:
                skip_connect = layers[i]

            layer_out = DownUnit(skip_connect=skip_connect, small_input=layers_td[i],
                                 big_input=layers_out[i-1], act='swish', **conv_kwargs)
            layers_out.append(layer_out)
        return layers_out


def PyramidalNeck(choice, num_filters, inputs_index):
    inputs_kwargs = dict(filters=num_filters,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='valid',
                         use_bias=False,
                         kernel_initializer=he_normal(seed=None))

    inputs = []
    f = []
    outputs = []
    for i in range(len(inputs_index)):
        index = inputs_index[i]
        size = FPN_INPUT_SIZE[index]
        filters = BACKBONE_OUT_FILTERS[choice][index]

        f_in = Input(shape=(size, size, filters))
        inputs.append(f_in)

        f_ = Unit(f_in, conv='conv', **inputs_kwargs)
        f.append(f_)

    f_ = Unit(inputs[-1], conv='conv', **inputs_kwargs)
    f_ = MaxPooling2D(pool_size=2, strides=2, padding='valid')(f_)
    f.append(f_)

    outputs.append(f[0])

    f = BiFPNBlock(f, num_filters=num_filters, idx=0, num_outputs=4)

    outputs.append(f[0])

    f = BiFPNBlock(f, num_filters=num_filters, idx=-1, num_outputs=1)

    outputs.append(f[0])

    model = Model(inputs=inputs, outputs=outputs, name='Neck')

    return model


if __name__ == '__main__':
    a = PyramidalNeck(3, 256, [2, 3, 4])
