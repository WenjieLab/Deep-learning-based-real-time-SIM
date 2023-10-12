# -*- coding: UTF-8 -*-
from keras import Input
from keras.layers import Conv2D, Activation, UpSampling2D, Lambda, Dropout, MaxPooling2D, multiply, add, Conv2DTranspose
from keras.layers import BatchNormalization
from keras import backend as K
from keras.models import Model
from common import fft2d, fftshift2d, gelu, pixel_shiffle, global_average_pooling2d
import tensorflow as tf
from keras.layers import Activation
# from qkeras import *
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


def npifft2d(input,gamma=0.1):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))

    a = tf.signal.ifft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(a) + 1e-8, gamma)
    # a1=np.fft.ifft2(a)
    # a2=np.abs(a1)
    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output

def FCALayer(input, channel, reduction=16):
    size_psc = input.get_shape().as_list()[1]
    # channel = input.get_shape().as_list()[3]
    # conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    # conv= Lambda(gelu)(conv)
    # conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    # conv= Lambda(gelu)(conv)
    absfft1 = Lambda(fft2d, arguments={'gamma': 0.8})(input)
    absfft1 = Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)
    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling2d)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    output = add([mul, input])
    return output

def AttnBlock2D(x, g, inter_channel, data_format='channels_last'):
 
    # theta_x = FCALayer(x,inter_channel,reduction=16)
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])

    return att_x


class Decoder(tf.keras.layers.Layer):

    def __init__(
        self,
        down_layer, layer, data_format,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.down_layer = down_layer
        self.layer = layer
        self.data_format = data_format

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'down_layer': self.down_layer,
            'layer': self.layer,
            'data_format': self.data_format,
        })
        return config

    def attention_up_and_concate(down_layer, layer, scale=2, data_format='channels_last'):

        if data_format == 'channels_last':
            in_channel = down_layer.get_shape().as_list()[3]
        else:
            in_channel = down_layer.get_shape().as_list()[1]

        up = Conv2DTranspose(in_channel, (4, 4), strides=(2, 2), padding='same', use_bias=False)(down_layer)

        if data_format == 'channels_last':
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        else:
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))  # 参考代码这个地方写错了，x[1] 写成了 x[3]
        concate = my_concat([up, layer])
        return concate


def attention_up_and_concate(down_layer, layer, scale=2, data_format='channels_last'):

    if data_format == 'channels_last':
        in_channel = down_layer.get_shape().as_list()[3]
    else:
        in_channel = down_layer.get_shape().as_list()[1]

    up = Conv2DTranspose(in_channel, (4, 4), strides=(2, 2), padding='same', use_bias=False)(down_layer)

    if data_format == 'channels_last':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))  # 参考代码这个地方写错了，x[1] 写成了 x[3]
    concate = my_concat([up, layer])
    return concate


def bn_attention_up_and_concate(down_layer, layer, scale=2, data_format='channels_last'):
    if data_format == 'channels_last':
        in_channel = down_layer.get_shape().as_list()[3]
    else:
        in_channel = down_layer.get_shape().as_list()[1]

    up = Conv2DTranspose(in_channel, (4, 4), strides=(2, 2), padding='same', use_bias=False)(down_layer)
    up = BatchNormalization(axis=-1, momentum=0.9)(up)

    if data_format == 'channels_last':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))  # 参考代码这个地方写错了，x[1] 写成了 x[3]
    concate = my_concat([up, layer])
    return concate


# Attention U-Net
def att_unet32(input_shape, size_psc=128, scale=2, data_format='channels_last'):
    # inputs = (3, 160, 160)
    inputs = Input(input_shape)
    depth = 4
    features = 32
    skips = []
    # conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
    # conv1 = Lambda(gelu)(conv)
    # x = FCALayer(conv1, features, reduction=16)
    # x = add([x, conv])
    conv = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Lambda(gelu)(conv)
    # depth = 0, 1, 2, 3

    # x=Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(inputs)
    # x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    # x2 = conv
    # x = Lambda(gelu)(x)

    for i in range(depth):
        # ENCODER
        x1 = Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = add([x, x1])
        # x = Conv2D(features, kernel_size=3, padding='same')(x)
        # x= Lambda(gelu)(x)
        # x=FCALayer(x,features,reduction=16)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_last')(x)
        features = features * 2

    # BOTTLENECK
    # x = Conv2D(features, kernel_size=3, padding='same')(x)
    # x = Lambda(gelu)(x)
    # x=FCALayer(x,features,reduction=16)
    x1 = Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = add([x, x1])
    # conv = Conv2D(features//2, kernel_size=3, padding='same')(inputs)
    # conv = Lambda(gelu)(conv)
    # x1 = FCALayer(conv, features//2, reduction=16)
    # x1 = add([x1, conv])
    # my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    # x= my_concat([x, x1])

    # DECODER
    for i in reversed(range(depth)):
        features = features // 2
        # x = Decoder.attention_up_and_concate(x, skips[i], data_format=data_format)
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x1 = Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = add([x, x1])
    n_label = 1
    # x=add([x2,x])
    # conv = Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(x)
    # conv = Lambda(gelu)(conv)
    # x = Lambda(pixel_shiffle, arguments={'scale': scale})(x)
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    # x = Lambda(pixel_shiffle, arguments={'scale': scale})(x)
    # conv6 = Conv2D(64, kernel_size=3, padding='same')(upsampled)
    # conv6 = Conv2D(1, kernel_size=1, padding='same')(x)
    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    # upsampled=UpSampling2D(size=(2, 2), data_format=data_format)(conv7)

    model = Model(inputs=inputs, outputs=conv7)
    return model

