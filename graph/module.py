import tensorflow as tf
import numpy as np
import sonnet as snt
#from parameters import Parameters
from tensorflow_utils import *

import sonnet as snt
## Code for Sonnet Modules

class ConvModule(snt.AbstractModule):
  def __init__(self,
               output_channels,
               kernel_shape=3,
               activation=tf.nn.relu,
               name='conv_module'):
    super(ConvModule, self).__init__(name='conv_module')
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._activation = activation

  def _build(self, inputs):
    with self._enter_variable_scope():
      conv = snt.Conv2D(self._output_channels, self._kernel_shape)
    return self._activation(conv(inputs))


class PerceptronModule(snt.AbstractModule):
  def __init__(self, hidden_size, activation=tf.nn.relu, name='perceptron_module'):
    super(PerceptronModule, self).__init__(name='perceptron_module')
    self._activation = activation
    self._hidden_size = hidden_size

  def _build(self, inputs):
    with self._enter_variable_scope():
      perceptron = snt.Linear(output_size=self._hidden_size, name='perceptron')
    return self._activation(perceptron(flatten_to_2d(inputs)))


class LinearModule(snt.AbstractModule):
  def __init__(self, hidden_size, name='linear_module', activation=None):
    super(LinearModule, self).__init__(name='linear_module')
    self._hidden_size = hidden_size

  def _build(self, inputs):
    with self._enter_variable_scope():
      linear_unit = snt.Linear(output_size=self._hidden_size, name='linear_unit')
    return linear_unit(flatten_to_2d(inputs))
