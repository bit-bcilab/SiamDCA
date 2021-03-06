

import tensorflow as tf
from keras import backend as K
from keras.layers import InputSpec, Layer, ZeroPadding2D
from keras import initializers, constraints, regularizers


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class DCNv2(Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 distribution='normal',
                 gain=1,
                 name=''):
        super(DCNv2, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias_attr = bias_attr

        self.conv_offset_padding = ZeroPadding2D(padding=((1, 0), (1, 0)))
        self.zero_padding = ZeroPadding2D(padding=((padding, padding+1), (padding, padding+1)))

    def build(self, input_shape):
        input_dim = self.input_dim
        filters = self.filters
        filter_size = self.filter_size
        bias_attr = self.bias_attr
        self.offset_w = self.add_weight('offset_w', shape=[filter_size, filter_size, input_dim, filter_size * filter_size * 3], initializer='zeros')
        self.offset_b = self.add_weight('offset_b', shape=[1, 1, 1, filter_size * filter_size * 3], initializer='zeros')
        self.dcn_weight = self.add_weight('dcn_weight', shape=[filters, input_dim, filter_size, filter_size], initializer='uniform')
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = self.add_weight('dcn_bias', shape=[filters, ], initializer='zeros')

    def compute_output_shape(self, input_shape):
        filters = self.filters
        batch, w, h = input_shape[:3]
        w = (w - self.filter_size + 2 * self.padding) // self.stride + 1
        h = (h - self.filter_size + 2 * self.padding) // self.stride + 1
        return (batch, w, h, filters)

    def call(self, inputs, **kwargs):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias

        # ???filter_size = 3, stride = 2, padding = 1?????? ??????padding2 = 'valid'???K.conv2d???????????????self.conv_offset_padding
        # ???filter_size = 3, stride = 1, padding = 1?????? ??????padding2 = 'same'???K.conv2d?????????????????????self.conv_offset_padding
        # ?????????????????????self.zero_padding???????????????????????????
        if stride == 2:
            temp = self.conv_offset_padding(inputs)
        else:
            temp = inputs
        padding2 = None
        if stride == 2:
            padding2 = 'valid'
        else:
            padding2 = 'same'
        offset_mask = K.conv2d(temp, self.offset_w, strides=(stride, stride), padding=padding2)
        offset_mask += self.offset_b

        offset_mask = tf.transpose(offset_mask, [0, 3, 1, 2])
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = tf.nn.sigmoid(mask)

        # ===================================
        N = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        out_C = tf.shape(dcn_weight)[0]
        in_C = tf.shape(dcn_weight)[1]
        kH = tf.shape(dcn_weight)[2]
        kW = tf.shape(dcn_weight)[3]
        W_f = tf.cast(W, tf.float32)
        H_f = tf.cast(H, tf.float32)
        kW_f = tf.cast(kW, tf.float32)
        kH_f = tf.cast(kH, tf.float32)

        out_W = (W_f + 2 * padding - (kW_f - 1)) // stride
        out_H = (H_f + 2 * padding - (kH_f - 1)) // stride
        out_W = tf.cast(out_W, tf.int32)
        out_H = tf.cast(out_H, tf.int32)
        out_W_f = tf.cast(out_W, tf.float32)
        out_H_f = tf.cast(out_H, tf.float32)

        # 1.????????????x??????????????????????????????pad_x
        pad_x = self.zero_padding(inputs)
        pad_x = tf.transpose(pad_x, [0, 3, 1, 2])

        # ?????????????????????pad_x????????????
        rows = tf.range(out_W_f, dtype=tf.float32) * stride + padding
        cols = tf.range(out_H_f, dtype=tf.float32) * stride + padding
        rows = tf.tile(rows[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis], [1, out_H, 1, 1, 1])
        cols = tf.tile(cols[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, out_W, 1, 1])
        start_pos_yx = tf.concat([cols, rows], axis=-1)  # [1, out_H, out_W, 1, 2]   ??????????????????????????????pad_x????????????
        start_pos_yx = tf.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   ??????????????????????????????pad_x????????????
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   ??????????????????????????????pad_x????????????
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   ??????????????????????????????pad_x????????????

        # ????????????????????????
        half_W = (kW_f - 1) / 2
        half_H = (kH_f - 1) / 2
        rows2 = tf.range(kW_f, dtype=tf.float32) - half_W
        cols2 = tf.range(kH_f, dtype=tf.float32) - half_H
        rows2 = tf.tile(rows2[tf.newaxis, :, tf.newaxis], [kH, 1, 1])
        cols2 = tf.tile(cols2[:, tf.newaxis, tf.newaxis], [1, kW, 1])
        filter_inner_offset_yx = tf.concat([cols2, rows2], axis=-1)  # [kH, kW, 2]   ????????????????????????
        filter_inner_offset_yx = tf.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   ????????????????????????
        filter_inner_offset_yx = tf.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   ????????????????????????
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   ????????????????????????
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   ????????????????????????

        mask = tf.transpose(mask, [0, 2, 3, 1])       # [N, out_H, out_W, kH*kW*1]
        offset = tf.transpose(offset, [0, 2, 3, 1])   # [N, out_H, out_W, kH*kW*2]
        offset_yx = tf.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # ????????????
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = tf.maximum(pos_y, 0.0)
        pos_y = tf.minimum(pos_y, H_f + padding * 2 - 1.0)
        pos_x = tf.maximum(pos_x, 0.0)
        pos_x = tf.minimum(pos_x, W_f + padding * 2 - 1.0)
        ytxt = tf.concat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = tf.transpose(pad_x, [0, 2, 3, 1])  # [N, pad_x_H, pad_x_W, C]

        mask = tf.reshape(mask, (N, out_H, out_W, kH, kW))  # [N, out_H, out_W, kH, kW]

        def _process_sample(args):
            _pad_x, _mask, _ytxt = args
            # _pad_x:    [pad_x_H, pad_x_W, in_C]
            # _mask:     [out_H, out_W, kH, kW]
            # _ytxt:     [out_H, out_W, kH*kW, 2]

            _ytxt = tf.reshape(_ytxt, (out_H * out_W * kH * kW, 2))  # [out_H*out_W*kH*kW, 2]
            _yt = _ytxt[:, :1]
            _xt = _ytxt[:, 1:]
            _y1 = tf.floor(_yt)
            _x1 = tf.floor(_xt)
            _y2 = _y1 + 1.0
            _x2 = _x1 + 1.0
            _y1x1 = tf.concat([_y1, _x1], -1)
            _y1x2 = tf.concat([_y1, _x2], -1)
            _y2x1 = tf.concat([_y2, _x1], -1)
            _y2x2 = tf.concat([_y2, _x2], -1)

            _y1x1_int = tf.cast(_y1x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v1 = tf.gather_nd(_pad_x, _y1x1_int)  # [out_H*out_W*kH*kW, in_C]
            _y1x2_int = tf.cast(_y1x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v2 = tf.gather_nd(_pad_x, _y1x2_int)  # [out_H*out_W*kH*kW, in_C]
            _y2x1_int = tf.cast(_y2x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v3 = tf.gather_nd(_pad_x, _y2x1_int)  # [out_H*out_W*kH*kW, in_C]
            _y2x2_int = tf.cast(_y2x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v4 = tf.gather_nd(_pad_x, _y2x2_int)  # [out_H*out_W*kH*kW, in_C]

            lh = _yt - _y1  # [out_H*out_W*kH*kW, 1]
            lw = _xt - _x1
            hh = 1 - lh
            hw = 1 - lw
            w1 = hh * hw
            w2 = hh * lw
            w3 = lh * hw
            w4 = lh * lw
            value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [out_H*out_W*kH*kW, in_C]
            _mask = tf.reshape(_mask, (out_H * out_W * kH * kW, 1))
            value = value * _mask
            value = tf.reshape(value, (out_H, out_W, kH, kW, in_C))
            value = tf.transpose(value, [0, 1, 4, 2, 3])   # [out_H, out_W, in_C, kH, kW]
            return value

        # ?????????????????????????????????????????????
        # new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
        # new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))   # [N, out_H, out_W, in_C * kH * kW]
        # new_x = tf.transpose(new_x, [0, 3, 1, 2])  # [N, in_C*kH*kW, out_H, out_W]
        # exp_new_x = tf.reshape(new_x, (N, 1, in_C*kH*kW, out_H, out_W))  # ??????1??????[N,      1, in_C*kH*kW, out_H, out_W]
        # reshape_w = tf.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))      # [1, out_C,  in_C*kH*kW,     1,     1]
        # out = exp_new_x * reshape_w                                   # ??????????????????[N, out_C,  in_C*kH*kW, out_H, out_W]
        # out = tf.reduce_sum(out, axis=[2, ])                           # ???2????????????[N, out_C, out_H, out_W]
        # out = tf.transpose(out, [0, 2, 3, 1])

        # ???????????????????????????1x1????????????????????????????????????
        new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
        new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))                # [N, out_H, out_W, in_C * kH * kW]
        tw = tf.transpose(dcn_weight, [1, 2, 3, 0])      # [out_C, in_C, kH, kW] -> [in_C, kH, kW, out_C]
        tw = tf.reshape(tw, (1, 1, in_C*kH*kW, out_C))   # [1, 1, in_C*kH*kW, out_C]  ??????1x1?????????
        out = K.conv2d(new_x, tw, strides=(1, 1), padding='valid')     # [N, out_H, out_W, out_C]
        return out


class DepthWiseCorr(Layer):
    """
    Limited by Keras and Tensorflow,
    the batch of Input() Layer or tf.placehodler() must be a definite number, rather than 'None'
    """
    def __init__(self, **kwargs):
        super(DepthWiseCorr, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        s1 = list(input_shape[0])
        s2 = list(input_shape[1])
        if s2[1] == 1:
            return input_shape[0]
        else:
            s = s1[1] - s2[1] + 1
            output_shape = [s1[0], s, s, s1[-1]]
            return tuple(output_shape)

    def call(self, inputs, **kwargs):
        x_batch = inputs[0]
        z_batch = inputs[1]
        batch = z_batch.shape[0].value

        x = x_batch[0, :, :, :]
        z = z_batch[0, :, :, :]
        x = x[None, :, :, :]
        z = z[:, :, :, None]
        out = K.depthwise_conv2d(x, z)

        if batch > 1:
            for i in range(batch-1):
                x = x_batch[i+1, :, :, :]
                z = z_batch[i+1, :, :, :]
                x = x[None, :, :, :]
                z = z[:, :, :, None]
                out_ = K.depthwise_conv2d(x, z)
                out = tf.concat([out, out_], axis=0)
        return out
