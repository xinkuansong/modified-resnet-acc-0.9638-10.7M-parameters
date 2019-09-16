"""
CIFAR10 datasets preparation.
"""
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np


def parse_dataset():
    # Import and parse CIFAR10 dataset
    subtract_pixel_mean = True
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return (x_train, y_train), (x_test, y_test)