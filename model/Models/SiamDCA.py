

from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Reshape
from keras.initializers import he_normal

from model.Backbone import Backbone
from model.Neck import PyramidalNeck
from model.DCA import DCA
from model.Layers import MatMul, Unit, WeightedAdd, Swish, LayerNormalization


def Build_SiamDCA(choice, x_shape, z_shape, fpn_in, num_filters,
                  num_att, hidden_dims, num_att_heads=8, d_k=32, d_v=32, dropout=0.1):
    if len(x_shape) == 3:
        x = Input(shape=x_shape)
        z = Input(shape=z_shape)
    else:
        x = Input(batch_shape=x_shape)
        z = Input(batch_shape=z_shape)
        z_shape = z_shape[1:]

    backbone = Backbone(z_shape, choice=choice, out_layers=fpn_in)
    neck = PyramidalNeck(choice=choice, num_filters=num_filters, inputs_index=fpn_in)
    xf = backbone(x)
    xf = neck(xf)
    zf = backbone(z)
    zf = neck(zf)

    dca = DCA(zf_shape=zf[0].shape.as_list()[1:], num_filters=num_filters, num_att=num_att,
              num_att_heads=num_att_heads, d_k=d_k, d_v=d_v, hidden_dims=hidden_dims, dropout=dropout)
    zf = dca(zf)

    head = EfficientHead(xf_shape=xf[0].shape.as_list()[1:], nz=zf[0].shape.as_list()[-1])
    cls, loc = head(xf + zf)

    model = Model([x, z], [cls, loc], name='SiamDCA')
    return model


def EfficientHead(xf_shape, nz):
    h, w, num_filters = xf_shape
    xf_ = []
    zf_spatial_ = []
    zf_channel_ = []
    cls_ = []
    loc_ = []
    for i in range(3):
        xf = Input(shape=(h, w, num_filters))
        zf_spatial = Input(shape=(num_filters, nz))
        zf_channel = Input(shape=(nz, num_filters))
        xf_.append(xf)
        zf_spatial_.append(zf_spatial)
        zf_channel_.append(zf_channel)

        cls, loc = Head(xf, zf_spatial, zf_channel, xf_shape=xf_shape)

        cls_.append(cls)
        loc_.append(loc)

    cls = WeightedAdd(layer_num=3, name='cls')(cls_)
    loc = WeightedAdd(layer_num=3, name='loc')(loc_)

    model = Model(xf_ + zf_spatial_ + zf_channel_, [cls, loc], name='Head')
    return model


def Head(xf, zf_spatial, zf_channel, xf_shape):
    h, w, num_filters = xf_shape
    conv_kwargs = dict(filters=num_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='valid',
                       use_bias=False,
                       kernel_initializer=he_normal(seed=None))
    loc_kwargs = conv_kwargs.copy()
    cls_kwargs = conv_kwargs.copy()

    xf_ = Reshape(target_shape=(-1, num_filters))(xf)

    score = MatMul()([xf_, zf_spatial])
    score = LayerNormalization(epsilon=1e-5)(score)
    score = Swish()(score)
    score = MatMul()([score, zf_channel])
    score = LayerNormalization(epsilon=1e-5)(score)
    score = Swish()(score)

    score = Reshape(target_shape=(h, w, num_filters))(score)

    score = Concatenate(axis=-1)([score, xf])
    score = Unit(score, conv='conv', act='swish', **conv_kwargs)

    cls = Unit(score, conv='conv', act='swish', **conv_kwargs)
    loc = Unit(score, conv='conv', act='swish', **conv_kwargs)

    cls_kwargs.update(dict(filters=2, use_bias=True))
    cls = Conv2D(**cls_kwargs)(cls)

    loc_kwargs.update(dict(filters=4, use_bias=True))
    loc = Conv2D(**loc_kwargs)(loc)
    return cls, loc


# if __name__ == '__main__':
#     a = Build_SiamDCA(3, [1, 256, 256, 3], [1, 128, 128, 3], fpn_in=[2, 3, 4], num_filters=64, num_att=1)
#     a.load_weights('weights/8-23-self/epoch39.h5')
#     pass
