from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.layer import *
from graph.sublayer import *
from graph.module import *

##############################################################################
## Code for Gatenet Graph construction
class Graph():

  def __init__(self, parameter_dict):
    self.gated_layers = None
    self.output_layer = None

    # Get the graph structure from given params
    self.graph_structure = self._graph_structure(parameter_dict)
    self.L = len(self.graph_structure)
    # Assume graph is a rectangular matrix (eg all layers have same # of
    # modules)
    self.M = len(self.graph_structure[0])


    # Define Hyperparameters
    self.C = parameter_dict['C'] # Number of output classes
    self.gamma = parameter_dict['gamma'] # Strength of gating
    self.sublayer_type = parameter_dict['sublayer_type']
    self.hidden_size = parameter_dict['hidden_size']

    # Build new graph
    self._build_graph()

  def _build_graph(self):
    with tf.variable_scope('init_graph'):
      ## Define Layers
      self.gated_layers = []

      for i in range(self.L):
        layer_structure = self.graph_structure[i] # List of modules
        gated_layer_defn = {'layer_structure': layer_structure,
                            'hidden_size': self.hidden_size,
                            'sublayer_type': self.sublayer_type}
        gated_layer = GatedLayer(gated_layer_defn, gamma=self.gamma, layer_name='gated_layer'+str(i))
        self.gated_layers.append(gated_layer)

      self.output_layer = OutputLayer(C=self.C, layer_name='output_layer')

  def _graph_structure(self, parameter_dict):
    '''Take the given parameter dictionary and '''
    ### Adapt to graph different definitions
    L = parameter_dict.get('L') # Number of layers
    M = parameter_dict.get('M') # Number of modules per layer
    module_type = parameter_dict.get('module_type')
    layer_structure = parameter_dict.get('layer_structure')
    graph_structure = parameter_dict.get('graph_structure')

    # If graph is defined by graph structure
    if graph_structure is not None:
      return graph_structure

    # If graph is defined by L & layer structure
    elif (L is not None) and (layer_structure is not None):
      return [layer_structure for _ in range(L)]

    # If graph is defined by L, M, & Module type:
    elif (L is not None) and (M is not None) and (module_type is not None):
      layer_structure = [module_type for _ in range(M)]
      return [layer_structure for _ in range(L)]

    else:
      print('Given graph parameters',
            '\nL', L,
            '\nM', M,
            '\nmodule_type', module_type,
            '\nlayer_structure', layer_structure,
            '\ngraph_structure', graph_structure)
      raise ValueError('The given graph definition is unsupported')

  def return_logits(self, input_images):
    with tf.name_scope('run_graph'):
      '''Return the output of graph based off of input_images'''
      next_input = input_images
      for i in range(self.L):
        next_input = self.gated_layers[i](next_input)
      logits = self.output_layer(next_input)
      return logits

  def determine_gates(self, image, x, sess):
    '''Output the values of the gates for image'''
    # TODO: Make gates work with varying # of modules per layer or
    # raise errors
    gates = np.zeros((self.L, self.M))
    for i in range(len(self.gated_layers)):
      g = sess.run([self.gated_layers[i].gates],
                   feed_dict={x:image})
      gates[i] = np.array(g)
    return gates

  def scopes(self, scope_type='modules'):
    scope_type = scope_type.lower() # To lowercase
    if (scope_type == 'modules') or (scope_type == 'm'):
      scopes = []
      for l in range(self.L):
        for mod in range(self.M):
          # Modules in gated layers
          scope = 'init_graph/gated_layer'+str(layer)+'/module_'+str(mod)
          scopes.append(scope)
        # Gates in gated layers
        scope = 'init_graph/gated_layer'+str(layer)+'/gates'
        scopes.append(scope)
      # Linear module in output layer
      scope = 'init_graph/output_layer/module'
      scopes.append(scope)
      return scopes
    elif (scope_type == 'layers') or (scope_type == 'l'):
      scopes = []
      for layer in range(self.L):
        # Gated layers
        scope = 'init_graph/gated_layer'+str(layer)
        scopes.append(scope)
      # Output layer
      scope = 'init_graph/output_layer'
      scopes.append(scope)
      return scopes
    elif (scope_type == 'graph') or (scope_type == 'g'):
      return ['init_graph']
    else:
      raise ValueError('The scope {} is not supported'.format(scope_type))
