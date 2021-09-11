

from keras.models import Model
from keras.layers import Input, MaxPooling2D, Concatenate
from keras.initializers import he_normal

from model.Backbone import BACKBONE_OUT_FILTERS
from model.Layers import Unit


def spp(x):
    x_1 = x
    x_2 = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
    x_3 = MaxPooling2D(pool_size=9, strides=1, padding='same')(x)
    x_4 = MaxPooling2D(pool_size=13, strides=1, padding='same')(x)
    out = Concatenate()([x_4, x_3, x_2, x_1])
    return out


def AdjustLayer(choice, inputs_index, num_filters, use_spp=False):
    inputs_kwargs = dict(filters=num_filters,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='valid',
                         use_bias=False,
                         kernel_initializer=he_normal(seed=None))

    inputs = []
    f = []
    for i in range(len(inputs_index)):
        filters = BACKBONE_OUT_FILTERS[choice][inputs_index[i]]

        f_in = Input(shape=(None, None, filters))
        inputs.append(f_in)

        if i == 0 and use_spp:
            f_ = spp(f_in)
            f_ = Unit(f_, conv='conv', **inputs_kwargs)
        else:
            f_ = Unit(f_in, conv='conv', **inputs_kwargs)
        f.append(f_)

    model = Model(inputs, f, name='adjust')
    return model
