

from keras.models import Model

from model.Backbone.EfficientNet import EfficientNetB0, EfficientNetB3
from model.Backbone.ResNet import ResNet50
from model.Backbone.ResNet50_SiamRPN import ResNet50_SiamRPN

PRETRAINED_BACKBONE_ROOT = 'weights/'

PRETRAINED_BACKBONE_PATH = ['ResNet50_SiamRPN.h5',
                            'ResNet50.h5',
                            'EfficientNet-B3.h5']

BACKBONE_OUT_LAYERS = [[5, 43, 88, 155, 189],
                       [5, 43, 88, 155, 189],
                       [26, 70, 114, 261, 378]]

BACKBONE_OUT_FILTERS = [[64, 256, 512, 1024, 2048],
                        [64, 256, 512, 1024, 2048],
                        [24, 32, 48, 136, 384]]

FPN_INPUT_SIZE = [128, 64, 32, 16, 8]

BACKBONES = [ResNet50_SiamRPN, ResNet50, EfficientNetB3]


def Backbone(input_shape, choice, out_layers):
    net = BACKBONES[choice](input_shape)
    outputs = []
    layers = BACKBONE_OUT_LAYERS[choice]
    for index in out_layers:
        outputs.append(net.layers[layers[index]].output)
    backbone = Model(net.inputs, outputs, name='Backbone')
    return backbone


# if __name__ == '__main__':
#     for i in range(len(BACKBONES)):
#         if i == 0:
#             input_shape = [127, 127, 3]
#         else:
#             input_shape = [128, 128, 3]
#         a = Backbone(input_shape, i, [2, 3, 4])
#     pass
