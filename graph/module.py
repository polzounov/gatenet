import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *


import sonnet as snt

######################################################################
## Code for Sonnet Modules

class ConvModule(snt.AbstractModule):
  def __init__(self, kernal_shape,output_channels,activation, name="conv_module"):
    super(ConvModule, self).__init__(name=name)
    self._kernal_shape = kernal_shape
    self._output_channels = output_channels
    self._activation = activation

  def _build(self, inputs):
      with self._enter_variable_scope():
          conv = snt.Conv2D(self._output_channels, self._kernal_shape)
          return self._activation(conv(inputs))



class PerceptronModule(snt.AbstractModule):
  def __init__(self, hidden_size, activation, name="perceptron_module"):
      super(PerceptronModule, self).__init__(name=name)
      self._activation = activation
      self._hidden_size = hidden_size

  def _build(self, inputs):
      with self._enter_variable_scope():
          perceptron = snt.Linear(output_size=self._hidden_size, name="perceptron")
          return self._activation(perceptron(inputs))


class LinearModule(snt.AbstractModule):
  def __init__(self, hidden_size, name="linear_module"):
      super(LinearModule, self).__init__(name=name)
      self._hidden_size = hidden_size

  def _build(self, inputs):
      with self._enter_variable_scope():
          linear_unit = snt.Linear(output_size=self._hidden_size, name="linear_unit")
          return linear_unit(inputs)


######################################################################
## Code for Modules
class Module():
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def process_module(self, inputTensor):
        pass


class PerceptronModule(Module):
    # 2d only
    def __init__(self, weights, biases, activation):
        super(PerceptronModule, self).__init__(weights, biases, activation)

    def process_module(self, input_tensor):
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases)


class ResidualPerceptronModule(Module):
    # 2d only
    def __init__(self, weights, biases, activation):
        super(ResidualPerceptronModule, self).__init__(weights, biases, activation)

    def process_module(self, input_tensor):
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases) + input_tensor


class LinearModule(Module):
    # 2d only
    def __init__(self, weights, biases, activation=None):
        super(LinearModule, self).__init__(weights, biases, activation)
        # Ignore the activation function if given

    def process_module(self, input_tensor):
        return tf.matmul(input_tensor, self.weights) + self.biases


class IdentityModule(Module):
    # Both 2d and 4d
    def __init__(self, *args):
        pass

    def process_module(self, input_tensor):
        return input_tensor


class ConvModule(Module):
    # 4D only
    def __init__(self, weights, activation, strides=[1, 1, 1, 1], padding='SAME'):
        super(ResidualConvModule, self).__init__(weights, biases, activation)
        self.strides = strides
        self.padding = padding

    def process_module(self, input_tensor):
        return self.activation(tf.nn.conv2d(input_tensor, 
                                            self.weights, 
                                            strides=self.strides, 
                                            padding=self.padding) + biases)


class ResidualConvModule(Module):
    # 4D only
    def __init__(self, weights, activation, strides=[1, 1, 1, 1], padding='SAME'):
        super(ResidualConvModule, self).__init__(weights, biases, activation)
        self.strides = strides
        self.padding = padding

    def process_module(self, input_tensor):
        return self.activation(tf.nn.conv2d(input_tensor, 
                                            self.weights, 
                                            strides=self.strides, 
                                            padding=self.padding) + biases) + input_tensor
