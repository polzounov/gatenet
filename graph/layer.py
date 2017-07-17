from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *
from graph.module import *

######################################################################
## Code for Layers
class Layer(snt.AbstractModule):
  def __init__(self, layer_definition, layer_name):
    super(Layer, self).__init__(name=layer_name)
    self.layer_structure = layer_definition.get('layer_structure')
    self.M = len(self.layer_structure)

    self.hidden_size = layer_definition.get('hidden_size') # Also # of output channels
    self.SublayerType = layer_definition.get('sublayer_type')
    self.act = layer_definition.get('activations', tf.nn.relu)
    self._init() # Init modules and sublayer in layer

  def _init(self):
    with self._enter_variable_scope():
      # Init modules
      self.modules = np.zeros(self.M, dtype=object)
      for i in range(self.M):
        ModuleType = self.layer_structure[i]
        self.modules[i] = ModuleType(self.hidden_size,
                                     activation=self.act,
                                     module_name='module_'+str(i))
      # Init sublayer
      self.sublayer = self.SublayerType(self.M, sublayer_name='sublayer')


class GatedLayer(Layer):
  def __init__(self, layer_definition, gamma=2.0, layer_name='gated_layer'):
    super(GatedLayer, self).__init__(layer_definition, layer_name=layer_name)
    with self._enter_variable_scope():
      self.gate_module = LinearModule(self.M, module_name='gates')
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

  def _build(self, input_tensors):
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

    return self.sublayer(output_tensors)


class OutputLayer(snt.AbstractModule):
  def __init__(self, C=10, layer_name='output_layer'):
    super(OutputLayer, self).__init__(name=layer_name)
    self.C = C
    self._init() # Init

  def _init(self):
    with self._enter_variable_scope():
      self.module = LinearModule(hidden_size=self.C, module_name='module')

  def _build(self, input_tensors):
    return self.module(input_tensors)
######################################################################