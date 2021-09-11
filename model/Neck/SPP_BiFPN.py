

from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate
from keras.initializers import he_normal, VarianceScaling

from model.Backbone import BACKBONE_OUT_FILTERS, FPN_INPUT_SIZE
from model.Layers import Unit, activation, UpSamplingAdd, WeightedAdd, DCNv2


def spp(x):
    x_1 = x
    x_2 = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
    x_3 = MaxPooling2D(pool_size=9, strides=1, padding='same')(x)
    x_4 = MaxPooling2D(pool_size=13, strides=1, padding='same')(x)
    out = Concatenate()([x_4, x_3, x_2, x_1])
    return out


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


def SPP_BiFPN(choice, inputs_index, outputs_index, num_filters, repeat_blocks, no_extra_down=False, use_spp=False, use_dcn=False):
    inputs_kwargs = dict(filters=num_filters,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='valid',
                         use_bias=False,
                         kernel_initializer=he_normal(seed=None))
    deformable_kwargs = dict(input_dim=num_filters,
                             filters=num_filters,
                             filter_size=3,
                             stride=1,
                             padding=1,
                             bias_attr=False)

    inputs = []
    f = []
    for i in range(len(inputs_index)):
        index = inputs_index[i]
        size = FPN_INPUT_SIZE[index]
        filters = BACKBONE_OUT_FILTERS[choice][index]

        f_in = Input(shape=(size, size, filters))
        inputs.append(f_in)

        if use_spp and i == 0:
            f_spp = spp(f_in)
            f_spp = Unit(f_spp, conv='conv', act='swish', **inputs_kwargs)
            f_spp = Unit(f_spp, conv='conv', **inputs_kwargs)
            f.append(f_spp)
        else:
            f_ = Unit(f_in, conv='conv', **inputs_kwargs)
            f.append(f_)

    if not no_extra_down:
        f_ = Unit(inputs[-1], conv='conv', **inputs_kwargs)
        f_ = MaxPooling2D(pool_size=2, strides=2, padding='valid')(f_)
        f.append(f_)

    for i in range(repeat_blocks):
        if i == repeat_blocks - 1:
            i = -1
        f = BiFPNBlock(f, num_filters=num_filters, idx=i, num_outputs=len(outputs_index))

    outputs = []
    for index in outputs_index:
        pos = inputs_index.index(index)
        output = f[pos]
        if use_dcn:
            output = Unit(output, conv='deformable', act='swish', **deformable_kwargs)
        outputs.append(output)

    model = Model(inputs=inputs, outputs=outputs, name='SPP-BiFPN')
    return model

