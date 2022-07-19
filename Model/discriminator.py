"""The file where we build our discriminator
as well as our discriminator"""

import tensorflow as tf

from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
    Conv2DTranspose, Input, Dropout,Add, LeakyReLU, Dense, Flatten, ReLU)
from tensorflow.keras.models import Model

from Model.layers import SelfAttention

def upsample(filters, size, apply_dropout=False):
    """upsampling block"""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result

def downsample(filters, size, apply_batchnorm=True):
    """down sampling block"""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(
        Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(
        Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))


    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def build_discriminator():
    """defines the discriminator"""
    kernal_size = 4


    model_input = Input(shape=[256, 256, 3], name='input_image')

    # Initial Res block
    x_1 = Conv2D(128, (kernal_size, kernal_size),
        strides=(2, 2),activation='relu', padding='same')(model_input)
    x_1 = Dropout(.075)(x_1)
    x_2 = Conv2D(128, (3, 3),activation='relu', padding='same')(x_1)
    x_2 = Conv2D(128, (3, 3),activation='relu', padding='same')(x_2)
    skip = Add()([x_1,x_2])

    # block 1
    x_3 = Dropout(.15)(skip)

    x_3 = SelfAttention(256)(x_3)
    x_3 = Conv2D(256, (kernal_size, kernal_size),
        strides=(2, 2),activation='relu', padding='same')(x_3)

    # block 2
    x_4 = Dropout(.15)(x_3)
    x_4 = Conv2D(512, (kernal_size, kernal_size),
        strides=(2, 2),activation='relu', padding='same')(x_4)

    # block 4
    x_5 = Dropout(.15)(x_4)
    x_5 = Conv2D(1024, (kernal_size, kernal_size),
        strides=(2, 2),activation='relu', padding='same')(x_5)

    # final block
    # x6 = SpectralNormalization(Conv2D(1, (kernal_size, kernal_size), activation='relu', padding='same'))(x_5)

    flat = Flatten()(x_5)

    pred = Dense(1,activation="sigmoid")(flat)

    return tf.keras.Model(inputs=model_input, outputs=pred)
