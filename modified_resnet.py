"""
Train a modified resnet on CIFAR10 dataset.
To filtering out redundant information and emphasizing higher layer features, modified resnet block is introduced.
"""

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Add
from keras.layers import Conv2D, BatchNormalization, Activation, Dense
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
import keras.backend as K
from numpy import pi
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical

## model blocks

def bn(x):

    x = BatchNormalization()(x)

    return x

def bn_activation(x):

    x = bn(x)
    x = Activation('relu')(x)

    return x

def conv2D(x, nb_filter):

    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer='he_uniform',
               padding='same',
               use_bias=False,
               kernel_regularizer=l2(1e-4))(x)

    return x

def transition(x, nb_filter):

    x = Conv2D(nb_filter, (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(1e-4))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = bn_activation(x)

    return x

def modified_residual_block(x, nb_layers, nb_filter):
    """
    :param x: inputs of modified residual block
    :param nb_layers:  number of conv2D layers in each block
    :param nb_filter: number of filter
    :return:
    """
    add_feat = x
    for i in range(nb_layers):

        x = conv2D(add_feat, nb_filter)
        x = bn_activation(x)

        add_feat = Add()([add_feat, x])
        add_feat = bn(add_feat)

    return add_feat

def modified_resnet(nb_classes, img_dim, depth, nb_blocks, nb_filter):

    inputs = Input(shape=img_dim)
    assert (depth - 4) % 3 == 0, "depth must be 3 * N + 4"

    # number of layers in each modified residual block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(1e-4))(inputs)
    x = bn_activation(x)

    # block 01
    x = modified_residual_block(x,
                                nb_layers,
                                nb_filter)

    x = transition(x, nb_filter)

    # block02
    x = modified_residual_block(x,
                                nb_layers,
                                nb_filter)

    x = transition(x, nb_filter)

    # block03
    x = modified_residual_block(x,
                                nb_layers,
                                nb_filter)

    # Final global-pooling and dense layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(nb_classes,
                    activation='softmax',
                    kernel_regularizer=l2(1e-4),
                    bias_regularizer=l2(1e-4))(x)

    modresnet = Model(inputs=[inputs], outputs=[outputs], name='Modified Residual Network')

    return modresnet

if __name__ == '__main__':

    nb_classes = 10
    input_shape = (32, 32, 3)
    depth = 19
    nb_blocks = 3
    nb_filter = 64

    model = modified_resnet(nb_classes=nb_classes, img_dim=input_shape, depth=depth, nb_blocks=nb_blocks, nb_filter=nb_filter)
    model.summary()

















