

from keras.models import Model
from keras.layers import Input, Lambda, Reshape
from keras.initializers import he_normal

from model.Layers import Unit, MultiHeadAttention, FeedForwardNet, Transpose


def DCA(zf_shape, num_filters, num_att, num_att_heads, d_k, d_v, hidden_dims, dropout):
    h = zf_shape[0]
    w = zf_shape[1]
    nz = (w // 2) * (h // 2)
    inputs = []
    zf_spatial_ = []
    zf_channel_ = []
    for i in range(3):
        # shape = (b, 16, 16, 256)
        zf_in = Input(shape=zf_shape)
        inputs.append(zf_in)
        # shape = (b, 8, 8, 256)
        zf = Lambda(lambda tensor: tensor[:, (h//4):(h - h//4), (w//4):(w - w//4), ...])(zf_in)
        # shape = (b, 256, 64), (b, 64, 256)
        zf_spatial, zf_channel = PreprocessZF(zf, nz, num_filters, num_att, num_att_heads, d_k, d_v, hidden_dims, dropout)
        zf_spatial_.append(zf_spatial)
        zf_channel_.append(zf_channel)
    model = Model(inputs=inputs, outputs=zf_spatial_ + zf_channel_)
    return model


def PreprocessZF(zf, nz, num_filters, num_att,
                 num_heads, d_k, d_v, hidden_dims, dropout):
    spatial_att = [MultiHeadAttention(num_heads=num_heads, model_dims=nz, seq_length=num_filters, d_k=d_k, d_v=d_v,
                                      dropout=dropout, use_bias=True, use_pe=True) for i in range(num_att)]
    spatial_ffn = [FeedForwardNet(seq_length=num_filters, model_dims=nz,
                                  hidden_dims=hidden_dims, dropout=dropout) for i in range(num_att)]
    channel_att = [MultiHeadAttention(num_heads=num_heads, model_dims=num_filters, seq_length=nz, d_k=d_k, d_v=d_v,
                                      dropout=dropout, use_bias=True, use_pe=True) for i in range(num_att)]
    channel_ffn = [FeedForwardNet(seq_length=nz, model_dims=num_filters,
                                  hidden_dims=hidden_dims, dropout=dropout) for i in range(num_att)]

    zf_kwargs = dict(filters=num_filters,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding='valid',
                     use_bias=False,
                     kernel_initializer=he_normal(seed=None))

    # shape = (b, 8, 8, 256)
    zf_spatial = Unit(zf, conv='conv', act='swish', **zf_kwargs)
    # shape = (b, 64, 256)
    zf_spatial = Reshape(target_shape=(-1, num_filters))(zf_spatial)
    # shape = (b, 256, 64)
    zf_spatial = Transpose(perm=(0, 2, 1))(zf_spatial)

    # shape = (b, 256, 64)
    for i in range(num_att):
        zf_spatial, _ = spatial_att[i]([zf_spatial, zf_spatial, zf_spatial])
        zf_spatial = spatial_ffn[i](zf_spatial)

    # shape = (b, 8, 8, 256)
    zf_channel = Unit(zf, conv='conv', act='swish', **zf_kwargs)
    # shape = (b, 64, 256)
    zf_channel = Reshape(target_shape=(-1, num_filters))(zf_channel)

    # shape = (b, 64, 256)
    for i in range(num_att):
        zf_channel, _ = channel_att[i]([zf_channel, zf_channel, zf_channel])
        zf_channel = channel_ffn[i](zf_channel)

    return zf_spatial, zf_channel
