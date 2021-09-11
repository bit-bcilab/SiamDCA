

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Input, BatchNormalization, ReLU, Add, ZeroPadding2D


def ResNet50_SiamRPN(inputs_shape, downsample=None, layer_config=None, used_layer=None):
    """
    由于两个tensor在互相卷积时，其batch数必须确定。因此，按照训练和测试的需求，在build model时设置不同的batch数。\n
    'train' mode时，batch数为2^n，'test' mode时默认为1。

    :param inputs_shape: tuple, input shape of backbone
    :param downsample: bool list, downsamle for each layer
    :param layer_config: int list, number of blocks in each layer
    :param used_layer: int list, index of chosen output layer
    """
    if used_layer is None:
        used_layer = [3, 4, 5]
        # used_layer = [4]
    if layer_config is None:
        layer_config = [3, 4, 6, 3]
    if downsample is None:
        downsample = [False, True, False, False]

    inputs = Input(shape=inputs_shape)
    outputs = Backbone(inputs=inputs, downsample=downsample,
                       layer_config=layer_config, used_layer=used_layer)
    model = Model(inputs=inputs, outputs=outputs, name='Backbone')
    # a = model.get_weights()
    # model.load_weights('weights/resnet50_SiamRPN.h5', by_name=True)
    # b = model.get_weights()
    return model


def Conv1(inputs):
    """
    Resnet50的第一卷积层
    """
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid',
                  use_bias=False, name='Conv1_conv')(inputs)
    bn = BatchNormalization(momentum=0.1, epsilon=1e-5, name='Conv1_bn')(conv)
    relu = ReLU(name='Conv1_relu')(bn)
    pad = ZeroPadding2D(padding=(1, 1))(relu)
    pool = MaxPool2D(pool_size=(3, 3), strides=2, name='Conv1_pool')(pad)
    return pool


def Residual(inputs, layer_index, downsample):
    """
    残差模块
    """
    inplanes = inputs.shape[-1].value

    if layer_index == 0:
        outplanes = 4 * inplanes
    else:
        outplanes = 2 * inplanes

    if downsample:
        conv_stride = (2, 2)
    else:
        conv_stride = (1, 1)

    if layer_index == 0:
        kernel_size = (1, 1)
    else:
        kernel_size = (3, 3)

    if layer_index > 1:
        if layer_index == 3:
            padding = (2, 2)
        elif layer_index == 2:
            padding = (1, 1)

        inputs = ZeroPadding2D(padding=padding)(inputs)
        conv = Conv2D(filters=outplanes, kernel_size=kernel_size, padding='valid', strides=conv_stride,
                      use_bias=False, dilation_rate=padding, name='Residual' + str(layer_index+2))(inputs)
    else:
        conv = Conv2D(filters=outplanes, kernel_size=kernel_size, padding='valid', strides=conv_stride,
                      use_bias=False, name='Residual' + str(layer_index+2))(inputs)

    bn = BatchNormalization(momentum=0.1, epsilon=1e-5, name='Residual' + str(layer_index+2) + '_bn')(conv)

    return bn


def Block(inputs, name, layer_index, block_index, downsample):
    """
    每个卷积层的Block
    """
    # 每层layer的第一个block的通道数是上一层输出的两倍
    if layer_index == 0 and block_index == 0:
        inplanes = inputs.shape[-1].value
    else:
        if block_index == 0:
            inplanes = int(inputs.shape[-1].value // 2)
        else:
            inplanes = int(inputs.shape[-1].value // 4)
    outplanes = int(4 * inplanes)

    padding = (1, 1)
    if layer_index > 1:
        if layer_index + block_index > 2:
            padding = (2, 2)
            if layer_index > 2 and block_index > 0:
                padding = (4, 4)

    conv1 = Conv2D(filters=inplanes, kernel_size=(1, 1), padding='same',
                   use_bias=False, name=name+'_conv1')(inputs)
    bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'_bn1')(conv1)
    bn1 = ReLU(name=name+'_relu1')(bn1)

    # 一般的ResNet50，每层layer的第一个block会通过设置步长为2来对上一层输入降采样
    # 但是SiamRPN的ResNet特征头中，有的层尺寸不下降，因此需要进行设置
    if block_index == 0 and downsample:
        conv_stride = (2, 2)
    else:
        conv_stride = (1, 1)
        bn1 = ZeroPadding2D(padding=padding)(bn1)

    conv2 = Conv2D(filters=inplanes, kernel_size=(3, 3),  strides=conv_stride,
                   use_bias=False, padding='valid', dilation_rate=padding, name=name+'_conv2')(bn1)

    bn2 = BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'_bn2')(conv2)
    bn2 = ReLU(name=name+'_relu2')(bn2)

    conv3 = Conv2D(filters=outplanes, kernel_size=(1, 1), padding='same',
                   use_bias=False, name=name+'_conv3')(bn2)
    bn3 = BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'_bn3')(conv3)

    return bn3


def Conv_layer(inputs, layer_config, layer_index, downsample):
    """
    卷积层
    """
    for block_index in range(layer_config[layer_index]):
        outputs = Block(inputs=inputs, name='Conv' + str(layer_index+2) + '_Block' + str(block_index+1),
                        layer_index=layer_index, block_index=block_index, downsample=downsample)

        # 每层的第一个Block使用带卷积的残差块，其他Block直接与输入相连作为残差即可
        if block_index == 0:
            residual = Residual(inputs=inputs, layer_index=layer_index, downsample=downsample)
        else:
            residual = inputs

        # 残差相加
        outputs = Add(name='Conv' + str(layer_index+2) + '_Block' + str(block_index+1) + 'Add')([outputs, residual])
        outputs = ReLU(name='Conv' + str(layer_index+2) + '_Block' + str(block_index+1) + '_relu3')(outputs)
        inputs = outputs

    return outputs


def Backbone(inputs, downsample, layer_config, used_layer):
    """
    Resnet50的变体
    """
    inputs = Conv1(inputs=inputs)
    layer_num = len(layer_config)
    out_num = len(used_layer)
    out = []

    used_layer = [used_layer[i] - 2 for i in range(out_num)]

    for layer_index in range(layer_num):
        outputs = Conv_layer(inputs=inputs, layer_config=layer_config,
                             layer_index=layer_index, downsample=downsample[layer_index])
        out.append(outputs)
        inputs = outputs

    out_ = [out[index] for index in used_layer]
    return out_


if __name__ == '__main__':
    model = ResNet50_SiamRPN((4, 255, 255, 3))
