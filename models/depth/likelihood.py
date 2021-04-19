import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D
from . import dropout

import functools
from edl.layers import Conv2DNormal

'''
Used to create models for testing likelihood loss functions(gaussian laplace, cacuhy etc)
sigma is True for adding the extra last layer 

'''
def create(input_shape, activation=tf.nn.relu, sigma=True, num_class=1):
    opts = locals().copy()
    model, opts = dropout.create(input_shape, drop_prob=0.1, sigma=sigma, activation=activation, num_class=num_class)
    return model, opts
