from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sonnet as snt
from tensorflow_utils import *


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    #return scale * tf.where(x >= 0.0, x, alpha * tf.exp(x) - alpha)
    return scale * tf.contrib.keras.activations.elu(x, alpha)

_nn_initializers = {
    'w': tf.contrib.keras.initializers.he_normal(),
    #'w': tf.contrib.keras.initializers.he_uniform(),
    #'w': tf.contrib.keras.initializers.glorot_normal(),
    #'w': tf.contrib.keras.initializers.glorot_uniform(),
    #"w": tf.random_normal_initializer(mean=0, stddev=0.01),
    'b': tf.random_normal_initializer(mean=0, stddev=0.01),
}

## Code for Sonnet Modules
class ConvModule(snt.AbstractModule):
  def __init__(self,
               output_channels,
               kernel_shape=3,
               activation=selu,
               module_name='conv_module'):
    super(ConvModule, self).__init__(name=module_name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._activation = activation

  def _build(self, inputs):
    conv = snt.Conv2D(self._output_channels,
                      self._kernel_shape,
                      initializers=_nn_initializers,
                      name='snt_conv_2d')
    return self._activation(conv(inputs))


class PerceptronModule(snt.AbstractModule):
  def __init__(self,
               hidden_size,
               activation=selu,
               module_name='perceptron_module'):
    super(PerceptronModule, self).__init__(name=module_name)
    self._activation = activation
    self._hidden_size = hidden_size

  def _build(self, inputs):
    perceptron = snt.Linear(output_size=self._hidden_size,
                            initializers=_nn_initializers,
                            name='snt_perceptron')
    return self._activation(perceptron(flatten_to_2d(inputs)))


class LinearModule(snt.AbstractModule):
  def __init__(self,
               hidden_size,
               activation=None,
               module_name='linear_module'):
    super(LinearModule, self).__init__(name=module_name)
    self._hidden_size = hidden_size

  def _build(self, inputs):
    linear_unit = snt.Linear(output_size=self._hidden_size,
                             name='snt_linear_unit')
    return linear_unit(flatten_to_2d(inputs))
