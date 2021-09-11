

import tensorflow as tf
from keras.layers import Layer
from keras.initializers import constant


class UpSamplingConcat(Layer):
    def __init__(self, **kwargs):
        super(UpSamplingConcat, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        s0 = list(input_shape[0])
        s1 = list(input_shape[1])
        s1[-1] += s0[-1]
        return tuple(s1)

    def call(self, inputs, **kwargs):
        small_inputs = inputs[0]
        big_inputs = inputs[1]
        big_shape = big_inputs.shape[1].value
        up_shape = (None, big_shape, big_shape, None)
        up_size = tf.shape(big_inputs)[1:3]
        up_inputs = tf.image.resize_bilinear(small_inputs, up_size)
        up_inputs.set_shape(up_shape)
        outputs = tf.concat([up_inputs, big_inputs], axis=-1)
        return outputs


class UpSamplingAdd(Layer):
    def __init__(self, **kwargs):
        super(UpSamplingAdd, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[-1]

    def build(self, input_shape):
        self.w = self.add_weight(name=self.name, shape=(2,), initializer=constant(1.), trainable=True, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        small_inputs = inputs[0]
        big_inputs = inputs[1]

        big_shape = big_inputs.shape[1].value
        up_shape = (None, big_shape, big_shape, None)
        up_size = tf.shape(big_inputs)[1:3]
        up_inputs = tf.image.resize_bilinear(small_inputs, up_size)
        up_inputs.set_shape(up_shape)

        weight = tf.nn.softmax(self.w)
        outputs = up_inputs * weight[0] + big_inputs * weight[1]
        return outputs


class WeightedAdd(Layer):
    """
    多层输出按权重相加 \n
    权重为可训练变量
    """
    def __init__(self, layer_num=3, **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)
        self.layer_num = layer_num

    def build(self, input_shape):
        self.w = self.add_weight(name='add', shape=(int(self.layer_num),), initializer='one', trainable=True)

    def get_config(self):
        base_config = super(WeightedAdd, self).get_config()
        base_config.update({'layer_num': self.layer_num})
        return base_config

    def call(self, inputs, **kwargs):
        weight = tf.nn.softmax(self.w)
        output = inputs[0] * weight[0]
        for i in range(1, self.layer_num):
            output += inputs[i] * weight[i]
        return output


class Squeeze(Layer):
    def __init__(self, axis=None, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        base_config = super(Squeeze, self).get_config()
        base_config.update({'axis': self.axis})
        return base_config

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        out_shape = []
        num_dims = len(shape)

        if isinstance(self.axis, list):
            axis = []
            for a in self.axis:
                if a == -1:
                    a = num_dims - 1
                axis.append(a)
            for d in range(num_dims):
                if shape[d] != 1 or d not in axis:
                    out_shape.append(shape[d])
        elif self.axis is None:
            for d in range(num_dims):
                if shape[d] != 1:
                    out_shape.append(shape[d])
        else:
            if self.axis == -1:
                axis = num_dims - 1
            else:
                axis = self.axis

            for d in range(num_dims):
                if shape[d] != 1 or d != axis:
                    out_shape.append(shape[d])

        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.squeeze(inputs, axis=self.axis)
        return outputs


class Stack(Layer):
    def __init__(self, axis, **kwargs):
        super(Stack, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        base_config = super(Stack, self).get_config()
        base_config.update({'axis': self.axis})
        return base_config

    def compute_output_shape(self, input_shape):
        num = len(input_shape)
        out_shape = list(input_shape[0])
        out_shape.insert(self.axis, num)
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.stack(inputs, axis=self.axis)
        return outputs


class Split(Layer):
    def __init__(self, axis, splits=None, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.splits = splits
        self.axis = axis

    def get_config(self):
        base_config = super(Split, self).get_config()
        base_config.update({'axis': self.axis, 'splits': self.splits})
        return base_config

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        num_channel = input_shape[self.axis]

        if isinstance(self.splits, list):
            count = 0
            for i in self.splits:
                count += i
            if count != num_channel:
                raise ValueError()
            num_split = len(self.splits)
            splits = self.splits
        else:
            if num_channel % self.splits != 0:
                raise ValueError()
            num_split = self.splits
            splits = [num_channel//num_split for i in range(num_split)]

        out_shape = []
        for i in range(num_split):
            s_ = shape
            s_[self.axis] = splits[i]
            out_shape.append(s_)
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.split(inputs, num_or_size_splits=self.splits, axis=self.axis)
        return outputs


class Tile(Layer):
    def __init__(self, multiples, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.multiples = multiples

    def get_config(self):
        base_config = super(Tile, self).get_config()
        base_config.update({'multiples': self.multiples})
        return base_config

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        out_shape = []
        for i in range(len(shape)):
            if shape[i] is not None:
                out_shape.append(int(shape[i] * self.multiples[i]))
            else:
                out_shape.append(shape[i])
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.tile(inputs, self.multiples)
        return outputs


class Transpose(Layer):
    def __init__(self, perm, **kwargs):
        super(Transpose, self).__init__(**kwargs)
        self.perm = perm

    def get_config(self):
        base_config = super(Transpose, self).get_config()
        base_config.update({'perm': self.perm})
        return base_config

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        out_shape = []
        for i in range(len(shape)):
            out_shape.append(shape[self.perm[i]])
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.transpose(inputs, perm=self.perm)
        return outputs


class MatMul(Layer):
    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        s0 = list(input_shape[0])
        s1 = list(input_shape[1])
        out_shape = s0[:-1]
        out_shape.append(s1[-1])
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.matmul(inputs[0], inputs[1])
        return outputs


class Center2Corner(Layer):
    def __init__(self, **kwargs):
        super(Center2Corner, self).__init__(**kwargs)
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        x1y1 = inputs[..., :2] - inputs[..., 2:] / 2.
        x2y2 = inputs[..., :2] + inputs[..., 2:] / 2.
        box = tf.concat([x1y1, x2y2], axis=-1)
        return box


class Corner2Center(Layer):
    def __init__(self, **kwargs):
        super(Corner2Center, self).__init__(**kwargs)
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        wh = inputs[..., 2:] - inputs[..., :2]
        xy = (inputs[..., :2] + inputs[..., 2:]) / 2.
        box = tf.concat([xy, wh], axis=-1)
        return box


class ROIAlign(Layer):
    def __init__(self, input_size, output_size, sample_ratio, num_roi, **kwargs):
        super(ROIAlign, self).__init__(**kwargs)
        self.output_size = output_size
        self.sample_ratio = sample_ratio
        self.num_roi = num_roi
        self.input_size = tf.constant([input_size[0] - 1., input_size[1] - 1.,
                                       input_size[0] - 1., input_size[1] - 1.], dtype=tf.float32)

    def get_config(self):
        base_config = super(ROIAlign, self).get_config()
        base_config.update({'input_size': self.input_size, 'output_size': self.output_size,
                            'sample_ratio': self.sample_ratio, 'num_roi': self.num_roi})
        return base_config

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        channel = input_shape[0][-1]
        out_shape = [batch, self.num_roi, self.output_size, self.output_size, channel]
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        feature_maps, boxes = inputs[0], inputs[1]
        batch = tf.cast(tf.shape(feature_maps)[0], dtype=tf.int64)

        box_indices = tf.range(0, limit=batch)
        box_indices = tf.reshape(box_indices, (batch, 1))
        box_indices = tf.tile(box_indices, (1, self.num_roi))
        box_indices = tf.reshape(box_indices, (-1, ))
        box_indices = tf.cast(box_indices, dtype=tf.int32)

        # 归一化
        boxes = tf.reshape(boxes, shape=(-1, 4))
        boxes = boxes / self.input_size
        x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
        x1 = tf.maximum(x1, 0.)
        y1 = tf.maximum(y1, 0.)
        x2 = tf.minimum(x2, 1.)
        y2 = tf.minimum(y2, 1.)

        bin_height = (y2 - y1) / self.output_size
        bin_width = (x2 - x1) / self.output_size

        grid_center_y1 = (y1 + 0.5 * bin_height / self.sample_ratio)
        grid_center_x1 = (x1 + 0.5 * bin_width / self.sample_ratio)
        grid_center_y2 = (y2 - 0.5 * bin_height / self.sample_ratio)
        grid_center_x2 = (x2 - 0.5 * bin_width / self.sample_ratio)

        new_boxes = tf.concat([grid_center_y1, grid_center_x1, grid_center_y2, grid_center_x2], axis=1)

        crop_size = tf.constant([self.output_size * self.sample_ratio, self.output_size * self.sample_ratio], dtype=tf.int32)
        sampled = tf.image.crop_and_resize(feature_maps, new_boxes, box_ind=box_indices, crop_size=crop_size, method='bilinear')
        aligned = tf.nn.avg_pool2d(sampled, self.sample_ratio, self.sample_ratio, padding='VALID')
        aligned = tf.reshape(aligned, shape=(batch, self.num_roi, self.output_size, self.output_size, -1))
        return aligned


class Box2ROI(Layer):
    def __init__(self, context_amount, **kwargs):
        super(Box2ROI, self).__init__(**kwargs)
        self.context_amount = context_amount

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(Box2ROI, self).get_config()
        base_config.update({'context_amount': self.context_amount})
        return base_config

    def call(self, inputs, **kwargs):
        # shape = (b, N, 2)
        roi_xy = inputs[..., :2]
        roi_wh = inputs[..., 2:]
        # shape = (b, N, 4)
        roi = tf.concat([roi_xy, (1. + self.context_amount) * roi_wh], axis=-1)
        return roi
