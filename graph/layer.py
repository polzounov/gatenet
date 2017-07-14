from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.module import *

######################################################################
## Code for Layers
class Layer():
  def __init__(self, layer_definition):
    self.M = layer_definition['M']
    self.hidden_size = layer_definition.get('hidden_size') # Also # of output channels
    # TODO: Make module type work with lists (for module variation inside layer)
    self.ModuleType = layer_definition.get('module_type')
    self.SublayerType = layer_definition.get('sublayer_type')
    self.act = layer_definition.get('activations', tf.nn.relu)
    self._build_modules() # Init modules in layer
    self._build_sublayer()

  def _build_modules(self):
    self.modules = np.zeros(self.M, dtype=object)
    for i in range(self.M):
      with tf.variable_scope('module_'+str(i)):
        self.modules[i] = self.ModuleType(self.hidden_size, activation=self.act)

  def _build_sublayer(self):
    with tf.variable_scope('sublayer'):
      self.sublayer = self.SublayerType(self.M)


class GatedLayer(Layer):
  def __init__(self, layer_definition, gamma=2.0):
    super(GatedLayer, self).__init__(layer_definition)
    with tf.variable_scope('gates'):
      self.gate_module = LinearModule(self.M)
    self.gates = None
    self.gamma = gamma

  def compute_gates(self, input_tensors):
    gates_unnormalized = self.gate_module(input_tensors)
    gates_pow = tf.pow(gates_unnormalized, self.gamma)

    gg = tf.reshape(tf.reduce_sum(gates_pow, axis = 1), [-1,1])
    num_cols = len(self.modules)
    gates_tiled = tf.tile(gg, [1, num_cols])

    gates_normalized = tf.nn.relu(gates_pow / gates_tiled) # why relu?
    self.gates = gates_normalized
    return gates_normalized

  def process_layer(self, input_tensors):
    gates = self.compute_gates(input_tensors)
    output_tensors = np.zeros(self.M, dtype=object)

    for i in range(self.M):
      # Get the number of rows in the fed value at run-time.
      output_tensors[i] = self.modules[i](input_tensors)
      tensor_shape = output_tensors[i].get_shape().as_list()

      # Temporary fix - TODO: fix this properly
      # if 4d / for convs - convert to 2d (to simplify gating)
      if len(tensor_shape) == 4:
        N,H,W,C = tensor_shape
        output_tensors[i] = tf.reshape(output_tensors[i], [-1, H*W*C])

      num_cols = np.int32(output_tensors[i].get_shape()[1])
      gg = tf.reshape(gates[:,i], [-1,1])
      gates_tiled = tf.tile(gg, [1,num_cols])
      output_tensors[i] = tf.multiply(output_tensors[i], gates_tiled)

      # if 4d / for convs - convert back
      if len(tensor_shape) == 4:
        output_tensors[i] = tf.reshape(output_tensors[i], [-1, H, W, C])

    return self.sublayer.process_sublayer(output_tensors)


class OutputLayer():
  def __init__(self, C=10):
    self.C = C
    self._build() # Init

  def _build(self):
    with tf.variable_scope('module'):
      self.module = LinearModule(hidden_size=self.C)

  def process_layer(self, input_tensors):
    return self.module(input_tensors)
######################################################################